"""
Module DataGenerator - điều phối trung tâm quá trình sinh dữ liệu tổng hợp.
Kết hợp FakerFactory, APIClient, ImageProcessor và StorageManager
để tạo ra các mẫu dữ liệu OCR hoàn chỉnh và nhất quán.
"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .api_client import APIClient
from .config import Config
from .faker_factory import FakerFactory
from .image_processor import ImageProcessor
from .quota_manager import QuotaManager, QuotaExceededError
from .rate_limiter import RateLimiter, RateLimitExceededError
from .storage import StorageManager, StorageError

logger = logging.getLogger(__name__)


class TemplateLoadError(Exception):
    """Lỗi khi không thể tải cấu hình template."""
    pass


class GenerationResult:
    """Kết quả sinh một mẫu dữ liệu đơn lẻ."""

    def __init__(
        self,
        success: bool,
        sample_id: str = "",
        doc_type: str = "",
        image_path: str = "",
        json_path: str = "",
        error_message: str = "",
        tokens_used: int = 0,
        elapsed_seconds: float = 0.0,
    ):
        self.success = success
        self.sample_id = sample_id
        self.doc_type = doc_type
        self.image_path = image_path
        self.json_path = json_path
        self.error_message = error_message
        self.tokens_used = tokens_used
        self.elapsed_seconds = elapsed_seconds


class DataGenerator:
    """
    Lớp điều phối trung tâm quá trình sinh dữ liệu tổng hợp.

    Phối hợp hoạt động của tất cả module:
        - FakerFactory: sinh dữ liệu văn bản giả
        - APIClient: lấy ảnh đại diện qua API
        - ImageProcessor: render ảnh tài liệu
        - StorageManager: lưu trữ kết quả
        - QuotaManager: kiểm soát hạn mức
        - RateLimiter: kiểm soát tốc độ gọi API
    """

    def __init__(
        self,
        config: Config,
        api_client: APIClient,
        storage_manager: StorageManager,
        image_processor: ImageProcessor,
        quota_manager: QuotaManager,
        rate_limiter: RateLimiter,
    ):
        """
        Khởi tạo DataGenerator với tất cả các thành phần phụ thuộc.

        Tham số:
            config: Cấu hình toàn hệ thống.
            api_client: Client kết nối API.
            storage_manager: Quản lý lưu trữ.
            image_processor: Xử lý ảnh.
            quota_manager: Quản lý hạn mức.
            rate_limiter: Kiểm soát tốc độ.
        """
        self.config = config
        self.api_client = api_client
        self.storage_manager = storage_manager
        self.image_processor = image_processor
        self.quota_manager = quota_manager
        self.rate_limiter = rate_limiter

        # Cache cấu hình template
        self._template_configs: Dict[str, Dict] = {}

        # Faker factory (hỗ trợ nhiều locale)
        self._faker_factory = FakerFactory(locale="en_AU")

        logger.info("Khoi tao DataGenerator.")

    def generate_batch(
        self,
        doc_type: str,
        count: int,
        progress_callback: Optional[Callable[[GenerationResult], None]] = None,
        num_workers: int = 1,
    ) -> List[GenerationResult]:
        """
        Sinh một lô mẫu dữ liệu cho một loại tài liệu.

        Hỗ trợ sinh tuần tự (num_workers=1) hoặc song song (num_workers>1).
        Khuyến nghị dùng num_workers=1 khi có giới hạn API chặt.

        Tham số:
            doc_type: Loại tài liệu cần sinh.
            count: Số mẫu cần sinh.
            progress_callback: Hàm gọi lại sau mỗi mẫu được sinh.
            num_workers: Số luồng song song.

        Trả về:
            Danh sách kết quả GenerationResult.
        """
        logger.info(
            "Bat dau sinh lo %d mau loai '%s' voi %d luong.",
            count, doc_type, num_workers,
        )

        # Tải cấu hình template
        template_config = self._load_template_config(doc_type)

        # Kiểm tra khả thi về quota
        feasibility = self.quota_manager.estimate_batch_feasibility(count)
        if not feasibility["kha_thi"]:
            logger.warning(
                "Chi co the thuc hien %d/%d mau do gioi han quota.",
                feasibility["co_the_thuc_hien"],
                count,
            )

        # Danh sách ID cần sinh
        sample_ids = [
            self.storage_manager.get_next_id(doc_type) for _ in range(count)
        ]

        results: List[GenerationResult] = []

        if num_workers <= 1:
            # Chế độ tuần tự - dễ gỡ lỗi hơn
            for sample_id in sample_ids:
                result = self._generate_one_safe(
                    doc_type, sample_id, template_config
                )
                results.append(result)
                if progress_callback:
                    progress_callback(result)
        else:
            # Chế độ song song với ThreadPoolExecutor
            results = self._generate_parallel(
                doc_type, sample_ids, template_config, progress_callback, num_workers
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
        """
        Sinh một mẫu dữ liệu đơn lẻ.

        Tham số:
            doc_type: Loại tài liệu.
            sample_id: ID cho mẫu này. Tự sinh nếu None.
            template_config: Cấu hình template (tải từ disk nếu None).

        Trả về:
            Đối tượng GenerationResult.
        """
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
    ) -> List[GenerationResult]:
        """Sinh dữ liệu song song sử dụng ThreadPoolExecutor."""
        results: List[GenerationResult] = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(
                    self._generate_one_safe, doc_type, sid, template_config
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
    ) -> GenerationResult:
        """
        Wrapper an toàn cho _generate_one: bắt tất cả ngoại lệ.

        Tham số:
            doc_type: Loại tài liệu.
            sample_id: ID mẫu.
            template_config: Cấu hình template.

        Trả về:
            GenerationResult luôn được trả về (không ném ngoại lệ).
        """
        start_time = time.monotonic()
        try:
            if self.config.dry_run:
                logger.debug("[DRY RUN] Bo qua viec sinh mau '%s'.", sample_id)
                time.sleep(0.01)  # Mô phỏng thời gian xử lý
                return GenerationResult(
                    success=True,
                    sample_id=sample_id,
                    doc_type=doc_type,
                    elapsed_seconds=time.monotonic() - start_time,
                )

            return self._generate_one(doc_type, sample_id, template_config)

        except QuotaExceededError as loi:
            logger.critical("Vuot gioi han quota: %s", loi)
            return GenerationResult(
                success=False,
                sample_id=sample_id,
                doc_type=doc_type,
                error_message=f"Vuot quota: {loi}",
                elapsed_seconds=time.monotonic() - start_time,
            )
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

    def _generate_one(
        self,
        doc_type: str,
        sample_id: str,
        template_config: Dict,
    ) -> GenerationResult:
        """
        Quy trình chính sinh một mẫu dữ liệu đầy đủ.

        Các bước:
            1. Kiểm tra quota.
            2. Sinh dữ liệu văn bản giả.
            3. Lấy ảnh đại diện (API hoặc placeholder).
            4. Render ảnh tài liệu.
            5. Lưu trữ kết quả.
            6. Ghi nhận tiêu thụ quota.

        Tham số:
            doc_type: Loại tài liệu.
            sample_id: ID mẫu.
            template_config: Cấu hình template đã tải.

        Trả về:
            GenerationResult với đầy đủ thông tin.
        """
        start_time = time.monotonic()
        estimated_tokens = self.quota_manager.estimate_tokens(
            text_content=str(template_config.get("fields", [])),
            image_request=self.config.image.avatar_use_api,
        )

        # Bước 1: Kiểm tra quota
        can_proceed, quota_msg = self.quota_manager.check_availability(estimated_tokens)
        if not can_proceed:
            raise QuotaExceededError(quota_msg)
        if quota_msg:
            logger.warning(quota_msg)

        # Bước 2: Sinh dữ liệu văn bản
        fields_config = template_config.get("fields", [])
        fields_data = self._faker_factory.generate_document_fields(fields_config)

        logger.debug(
            "Da sinh du lieu cho mau '%s': %d truong.", sample_id, len(fields_data)
        )

        # Bước 3: Lấy ảnh đại diện
        avatar_image: Optional[Image.Image] = None
        avatar_config = template_config.get("avatar", {})
        tokens_used = 0

        if avatar_config.get("enabled", False):
            avatar_size = avatar_config.get("size", [150, 190])
            gender = fields_data.get("gender", "unknown")
            name_seed = fields_data.get("ho_ten", fields_data.get("full_name", sample_id))

            avatar_image = self.api_client.get_avatar(
                seed=name_seed,
                width=avatar_size[0],
                height=avatar_size[1],
                use_api=self.config.image.avatar_use_api,
                gender=gender,
            )

            if self.config.image.avatar_use_api:
                tokens_used += 500  # Ước tính token cho yêu cầu ảnh

        # Bước 4: Render ảnh tài liệu
        rendered_image, bounding_boxes = self.image_processor.render_document(
            doc_type=doc_type,
            template_config=template_config,
            fields_data=fields_data,
            avatar_image=avatar_image,
        )

        # Bước 5: Lưu trữ
        metadata = {
            "locale": self._faker_factory.locale,
            "api_platform": self.config.api.platform,
            "augmentation_enabled": self.config.image.enable_augmentation,
        }

        image_path, json_path = self.storage_manager.save_sample(
            doc_type=doc_type,
            sample_id=sample_id,
            image=rendered_image,
            fields_data=fields_data,
            bounding_boxes=bounding_boxes,
            metadata=metadata,
        )

        # Bước 6: Ghi nhận tiêu thụ quota
        self.quota_manager.record_usage(tokens_used=tokens_used, success=True)

        elapsed = time.monotonic() - start_time
        quota_status = self.quota_manager.get_status()

        logger.debug(
            "Hoan thanh mau '%s' trong %.2f giay.",
            sample_id, elapsed,
        )

        return GenerationResult(
            success=True,
            sample_id=sample_id,
            doc_type=doc_type,
            image_path=str(image_path),
            json_path=str(json_path),
            tokens_used=tokens_used,
            elapsed_seconds=elapsed,
        )

    def _load_template_config(self, doc_type: str) -> Dict:
        """
        Tải cấu hình template từ tệp base.json.
        Kết quả được cache để tránh đọc lại nhiều lần.

        Tham số:
            doc_type: Loại tài liệu cần tải.

        Trả về:
            Từ điển cấu hình template.

        Ném ra:
            TemplateLoadError: Nếu không tìm thấy hoặc không đọc được tệp.
        """
        if doc_type in self._template_configs:
            return self._template_configs[doc_type]

        base_json_path = (
            Path(self.config.storage.templates_dir) / doc_type / "base.json"
        )

        if not base_json_path.exists():
            raise TemplateLoadError(
                f"Khong tim thay tep cau hinh template: '{base_json_path}'. "
                f"Vui long tao tep base.json trong thu muc templates/{doc_type}/."
            )

        try:
            with open(base_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self._template_configs[doc_type] = config
            logger.info(
                "Da tai cau hinh template '%s': %d truong.",
                doc_type,
                len(config.get("fields", [])),
            )
            return config

        except json.JSONDecodeError as loi:
            raise TemplateLoadError(
                f"Tep base.json bi hong (khong phai JSON hop le): '{base_json_path}'. Chi tiet: {loi}"
            ) from loi
        except OSError as loi:
            raise TemplateLoadError(
                f"Khong the doc tep '{base_json_path}': {loi}"
            ) from loi

    def validate_template(self, doc_type: str) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của cấu hình template.

        Tham số:
            doc_type: Loại tài liệu cần kiểm tra.

        Trả về:
            Tuple (hop_le, danh_sach_loi).
        """
        errors = []
        try:
            config = self._load_template_config(doc_type)
        except TemplateLoadError as loi:
            return False, [str(loi)]

        # Kiểm tra các trường bắt buộc
        required_keys = ["document_type", "fields"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Thieu truong bat buoc: '{key}'.")

        # Kiểm tra từng trường
        for i, field in enumerate(config.get("fields", [])):
            if "key" not in field:
                errors.append(f"Truong thu {i + 1} thieu 'key'.")
            if "faker_type" not in field:
                errors.append(f"Truong '{field.get('key', i)}' thieu 'faker_type'.")
            if "position" not in field:
                errors.append(f"Truong '{field.get('key', i)}' thieu 'position'.")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Kiem tra template '%s': hop le.", doc_type)
        else:
            logger.warning(
                "Kiem tra template '%s': %d loi phat hien.", doc_type, len(errors)
            )

        return is_valid, errors

    def get_generation_summary(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """
        Tổng hợp thống kê từ danh sách kết quả.

        Tham số:
            results: Danh sách kết quả từ generate_batch.

        Trả về:
            Từ điển thống kê tổng hợp.
        """
        if not results:
            return {}

        success_results = [r for r in results if r.success]
        failure_results = [r for r in results if not r.success]

        elapsed_times = [r.elapsed_seconds for r in success_results]
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
        total_tokens = sum(r.tokens_used for r in results)

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
            "tong_token_da_dung": total_tokens,
            "phan_loai_loi": failure_msgs,
        }
