# src/generator.py
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .api_client import APIClient
from .config import Config
from .faker_factory import FakerFactory
from .image_processor import ImageProcessor
from .rate_limiter import RateLimiter, RateLimitExceededError
from .storage import StorageManager, StorageError

logger = logging.getLogger(__name__)

class TemplateLoadError(Exception):
    pass

class GenerationResult:
    def __init__(
        self,
        success: bool,
        sample_id: str = "",
        doc_type: str = "",
        image_path: str = "",
        json_path: str = "",
        error_message: str = "",
        elapsed_seconds: float = 0.0,
    ):
        self.success = success
        self.sample_id = sample_id
        self.doc_type = doc_type
        self.image_path = image_path
        self.json_path = json_path
        self.error_message = error_message
        self.elapsed_seconds = elapsed_seconds

class DataGenerator:
    def __init__(
        self,
        config: Config,
        api_client: APIClient,
        storage_manager: StorageManager,
        image_processor: ImageProcessor,
        rate_limiter: RateLimiter,
    ):
        self.config = config
        self.api_client = api_client
        self.storage_manager = storage_manager
        self.image_processor = image_processor
        self.rate_limiter = rate_limiter

        self._template_configs: Dict[str, Dict] = {}
        self._faker_factory = FakerFactory(locale="en_AU")

        logger.info("Khoi tao DataGenerator.")

    def generate_batch(
        self,
        doc_type: str,
        count: int,
        progress_callback: Optional[Callable[[GenerationResult], None]] = None,
        num_workers: int = 1,
        state=None,
    ) -> List[GenerationResult]:
        logger.info(
            "Bat dau sinh lo %d mau loai '%s' voi %d luong.",
            count, doc_type, num_workers,
        )

        template_config = self._load_template_config(doc_type, state)

        sample_ids = [
            self.storage_manager.get_next_id(doc_type) for _ in range(count)
        ]

        results: List[GenerationResult] = []

        if num_workers <= 1:
            for sample_id in sample_ids:
                result = self._generate_one_safe(
                    doc_type, sample_id, template_config,state
                )
                results.append(result)
                if progress_callback:
                    progress_callback(result)
        else:
            results = self._generate_parallel(
                doc_type, sample_ids, template_config, progress_callback, num_workers, state
            )

        success_count = sum(1 for r in results if r.success)
        logger.info(
            "Hoan thanh lo sinh du lieu: %d/%d thanh cong.",
            success_count, count,
        )
        return results

    def generate_single(
        self,
        doc_type: str,
        sample_id: Optional[str] = None,
        template_config: Optional[Dict] = None,
    ) -> GenerationResult:
        if template_config is None:
            template_config = self._load_template_config(doc_type)
        if sample_id is None:
            sample_id = self.storage_manager.get_next_id(doc_type)

        return self._generate_one_safe(doc_type, sample_id, template_config)

    def _generate_parallel(
        self,
        doc_type: str,
        sample_ids: List[str],
        template_config: Dict,
        progress_callback: Optional[Callable],
        num_workers: int,
        state=None,
    ) -> List[GenerationResult]:
        results: List[GenerationResult] = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(
                    self._generate_one_safe, doc_type, sid, template_config,  state
                ): sid
                for sid in sample_ids
            }

            for future in as_completed(future_to_id):
                sample_id = future_to_id[future]
                try:
                    result = future.result()
                except Exception as loi:
                    logger.error(
                        "Loi khong xac dinh khi sinh mau '%s': %s", sample_id, loi
                    )
                    result = GenerationResult(
                        success=False,
                        sample_id=sample_id,
                        doc_type=doc_type,
                        error_message=str(loi),
                    )
                results.append(result)
                if progress_callback:
                    progress_callback(result)

        return results

    def _generate_one_safe(
        self,
        doc_type: str,
        sample_id: str,
        template_config: Dict,
        state=None,
    ) -> GenerationResult:
        start_time = time.monotonic()
        try:
            if self.config.dry_run:
                logger.debug("[DRY RUN] Bo qua viec sinh mau '%s'.", sample_id)
                time.sleep(0.01)
                return GenerationResult(
                    success=True,
                    sample_id=sample_id,
                    doc_type=doc_type,
                    elapsed_seconds=time.monotonic() - start_time,
                )

            return self._generate_one(doc_type, sample_id, template_config, state)

        except StorageError as loi:
            logger.error("Loi luu tru mau '%s': %s", sample_id, loi)
            return GenerationResult(
                success=False,
                sample_id=sample_id,
                doc_type=doc_type,
                error_message=f"Loi luu tru: {loi}",
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as loi:
            logger.exception("Loi bat ngo khi sinh mau '%s': %s", sample_id, loi)
            return GenerationResult(
                success=False,
                sample_id=sample_id,
                doc_type=doc_type,
                error_message=str(loi),
                elapsed_seconds=time.monotonic() - start_time,
            )

    def _parse_custom_json(self, data: Any) -> Any:
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                k_lower = k.lower()
                
                if isinstance(v, (dict, list)):
                    result[k] = self._parse_custom_json(v)
                elif any(word in k_lower for word in ["first_name", "given_name"]):
                    result[k] = self._faker_factory.faker.first_name().upper()
                elif any(word in k_lower for word in ["middle_name"]):
                    result[k] = self._faker_factory.faker.first_name().upper() if v else None
                elif any(word in k_lower for word in ["last_name", "surname"]):
                    result[k] = self._faker_factory.faker.last_name().upper()
                elif any(word in k_lower for word in ["dob", "birth"]):
                    if isinstance(v, str) and "-" not in v and v.isdigit():
                        result[k] = self._faker_factory.faker.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d%m%Y")
                    else:
                        result[k] = self._faker_factory.faker.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d-%m-%Y")
                elif any(word in k_lower for word in ["expiry", "expiration"]):
                    result[k] = self._faker_factory.faker.date_between(start_date="today", end_date="+5y").strftime("%d-%m-%Y")
                elif "line_1" in k_lower:
                    result[k] = f"UNIT {self._faker_factory.faker.random_int(min=1, max=99)}"
                elif any(word in k_lower for word in ["line_2", "street"]):
                    result[k] = self._faker_factory.faker.street_address().upper()
                elif any(word in k_lower for word in ["suburb", "city", "town"]):
                    result[k] = self._faker_factory.faker.city().upper()
                elif any(word in k_lower for word in ["postcode", "zip"]):
                    result[k] = str(self._faker_factory.faker.postcode())
                elif any(word in k_lower for word in ["licence_number", "license_number"]):
                    result[k] = str(self._faker_factory.faker.random_number(digits=9, fix_len=True))
                elif "card_number" in k_lower:
                    result[k] = f"D{self._faker_factory.faker.random_number(digits=7, fix_len=True)}"
                elif "barcode" in k_lower:
                    result[k] = f"ABnoteNZ{self._faker_factory.faker.random_number(digits=10, fix_len=True)}"
                else:
                    result[k] = v
            return result
        elif isinstance(data, list):
            return [self._parse_custom_json(i) for i in data]
        return data
    
    def _generate_one(self, doc_type: str, sample_id: str, template_config: Dict, state=None) -> GenerationResult:
        start_time = time.monotonic()
        
        raw_json = template_config 
        nested_json_data = self._parse_custom_json(raw_json)

        template_image = self.image_processor._load_template_image(doc_type, template_config, state)

        api_image = self.api_client.generate_document(template_image, nested_json_data)
        
        if not api_image:
            raise Exception("API khong phan hoi anh.")

        img_p, json_p = self.storage_manager.save_sample(
            doc_type=doc_type, sample_id=sample_id, image=api_image, 
            fields_data=nested_json_data, bounding_boxes=[], metadata={},state=state,
        )

        return GenerationResult(True, sample_id, doc_type, str(img_p), str(json_p), "", time.monotonic()-start_time)
        
    def _load_template_config(self, doc_type: str, state: str = None) -> dict:
        cache_key = f"{doc_type}_{state}" if state else doc_type
        if cache_key in self._template_configs:
            return self._template_configs[cache_key]
    
        if state:
            base_json_path = Path(self.config.storage.templates_dir) / doc_type / state / "base.json"
        else:
            base_json_path = Path(self.config.storage.templates_dir) / doc_type / "base.json"

        if not base_json_path.exists():
            raise TemplateLoadError(
                f"Khong tim thay tep cau hinh template: '{base_json_path}'."
            )

        try:
            with open(base_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self._template_configs[doc_type] = config
            logger.info("Da tai cau hinh template '%s'", doc_type)
            return config

        except json.JSONDecodeError as loi:
            raise TemplateLoadError(f"Tep base.json bi hong: {loi}") from loi
        except OSError as loi:
            raise TemplateLoadError(f"Khong the doc tep: {loi}") from loi

    def validate_template(self, doc_type: str) -> Tuple[bool, List[str]]:
        errors = []
        try:
            config = self._load_template_config(doc_type)
        except TemplateLoadError as loi:
            return False, [str(loi)]

        required_keys = ["document_type", "fields"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Thieu truong bat buoc: '{key}'.")

        for i, field in enumerate(config.get("fields", [])):
            if "key" not in field:
                errors.append(f"Truong thu {i + 1} thieu 'key'.")
            if "faker_type" not in field:
                errors.append(f"Truong '{field.get('key', i)}' thieu 'faker_type'.")
            if "position" not in field:
                errors.append(f"Truong '{field.get('key', i)}' thieu 'position'.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_generation_summary(self, results: List[GenerationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        success_results = [r for r in results if r.success]
        failure_results = [r for r in results if not r.success]

        elapsed_times = [r.elapsed_seconds for r in success_results]
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

        failure_msgs = {}
        for r in failure_results:
            msg_key = r.error_message[:50] if r.error_message else "Khong ro"
            failure_msgs[msg_key] = failure_msgs.get(msg_key, 0) + 1

        return {
            "tong_yeu_cau": len(results),
            "thanh_cong": len(success_results),
            "that_bai": len(failure_results),
            "ti_le_thanh_cong_phan_tram": round(
                len(success_results) / len(results) * 100, 2
            ),
            "thoi_gian_trung_binh_giay": round(avg_time, 3),
            "phan_loai_loi": failure_msgs,
        }