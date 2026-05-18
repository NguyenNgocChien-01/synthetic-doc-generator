# src/generator.py
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
# from rich.prompt import result

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

    # def _parse_custom_json(self, data: Any, _shared: dict = None) -> Any:
    #     """
    #     _shared: dict dùng để chia sẻ các giá trị cần nhất quán giữa các sub-dict:
    #     - dob_dd, dob_mm, dob_yyyy  → từ lần đầu gặp dob/date_of_birth
    #     - expiry_date               → từ lần đầu gặp expiry/expiration
    #     """
    #     if _shared is None:
    #         _shared = {}   # root call tạo mới

    #     if isinstance(data, dict):
    #         result = {}

    #         _dob_dd = _dob_mm = _dob_yyyy = None

    #         for k, v in data.items():
    #             k_lower = k.lower()
    #             if any(w in k_lower for w in ["dob", "date_of_birth", "birth"]) \
    #                     and "watermark" not in k_lower:

    #                 if "dob_dd" in _shared:
    #                     _dob_dd = _shared["dob_dd"]
    #                     _dob_mm = _shared["dob_mm"]
    #                     _dob_yyyy = _shared["dob_yyyy"]
    #                 else:
    #                     dob_obj = self._faker_factory.faker.date_of_birth(
    #                         minimum_age=18, maximum_age=70
    #                     )
    #                     _dob_dd  = dob_obj.strftime("%d")
    #                     _dob_mm  = dob_obj.strftime("%m")
    #                     _dob_yyyy = dob_obj.strftime("%Y")
    #                     _shared["dob_dd"]   = _dob_dd
    #                     _shared["dob_mm"]   = _dob_mm
    #                     _shared["dob_yyyy"] = _dob_yyyy

    #                 # format output
    #                 v_str = str(v).strip() if v else ""
    #                 if v_str.isdigit() and "-" not in v_str:
    #                     result[k] = f"{_dob_dd}{_dob_mm}{_dob_yyyy}"
    #                 elif "-" in v_str and len(v_str) == 10:
    #                     result[k] = f"{_dob_yyyy}-{_dob_mm}-{_dob_dd}"   # ISO
    #                 else:
    #                     result[k] = f"{_dob_dd}-{_dob_mm}-{_dob_yyyy}"


    #         for k, v in data.items():
    #             k_lower = k.lower()

    #             if k in result:
    #                 continue   

    #             # ── nested dict/list ──────────────────────────────────────────
    #             if k_lower == "conditions_legend":
    #                 result[k] = v

    #             elif isinstance(v, (dict, list)) and k_lower not in ["members", "cardholders"]:
    #                 result[k] = self._parse_custom_json(v, _shared)

    #             elif k_lower in ["members", "cardholders"] and isinstance(v, list):
    #                 import random, string
    #                 num_members = random.randint(1, 5)
    #                 new_members = []
    #                 for i in range(1, num_members + 1):
    #                     first  = (random.choice(string.ascii_uppercase)
    #                             if random.random() < 0.1
    #                             else self._faker_factory.faker.first_name().upper())
    #                     last   = (random.choice(string.ascii_uppercase)
    #                             if random.random() < 0.1
    #                             else self._faker_factory.faker.last_name().upper())
    #                     middle = (random.choice(string.ascii_uppercase)
    #                             if random.random() > 0.3 else None)
    #                     parts  = [str(i), first] + ([middle] if middle else []) + [last]
    #                     new_members.append({
    #                         "position": i, "first_name": first,
    #                         "middle_initial": middle, "last_name": last,
    #                         "full_name": " ".join(parts)
    #                     })
    #                 result[k] = new_members

    #             # ── medicare ──────────────────────────────────────────────────
    #             elif k_lower == "medicare_card_number":
    #                 p1 = self._faker_factory.faker.random_number(digits=4, fix_len=True)
    #                 p2 = self._faker_factory.faker.random_number(digits=5, fix_len=True)
    #                 p3 = self._faker_factory.faker.random_number(digits=1)
    #                 result[k] = f"{p1} {p2} {p3}"

    #             # ── name fields ───────────────────────────────────────────────
    #             elif any(w in k_lower for w in ["first_name", "given_name"]):
    #                 result[k] = self._faker_factory.faker.first_name().upper()

    #             elif "middle_name" in k_lower:
    #                 result[k] = self._faker_factory.faker.first_name().upper() if v else None

    #             elif any(w in k_lower for w in ["last_name", "surname", "family_name"]):
    #                 result[k] = self._faker_factory.faker.last_name().upper()

    #             # ── watermark (dob compact) ───────────────────────────────────
    #             elif "watermark" in k_lower:
    #                 dd  = _dob_dd  or _shared.get("dob_dd",  "")
    #                 mm  = _dob_mm  or _shared.get("dob_mm",  "")
    #                 yyyy = _dob_yyyy or _shared.get("dob_yyyy", "")
    #                 result[k] = f"{dd}{mm}{yyyy}" if dd else v

    #             # ── age_indicator: MM-YY────────────────────
    #             elif "age_indicator" in k_lower:
    #                 mm   = _dob_mm   or _shared.get("dob_mm",   "")
    #                 yyyy = _dob_yyyy or _shared.get("dob_yyyy", "")
    #                 result[k] = f"{mm}-{yyyy[-2:]}" if mm and yyyy else v

    #             # ── expiry / expiration ───────────────────────────────────────
    #             elif any(w in k_lower for w in ["expiry", "expiration"]):
    #                 if "expiry_date" not in _shared:
    #                     _shared["expiry_date"] = self._faker_factory.faker.date_between(
    #                         start_date="+1y", end_date="+5y"
    #                     )
    #                 fake_date = _shared["expiry_date"]

    #                 v_str = str(v).strip()
    #                 if "/" in v_str:
    #                     if v_str.count("/") == 2 or len(v_str) > 7:
    #                         result[k] = fake_date.strftime("%d/%m/%Y")
    #                     else:
    #                         result[k] = fake_date.strftime("%m/%Y")
    #                 elif "-" in v_str:
    #                     if len(v_str) == 10 and (v_str.startswith("20") or v_str.startswith("19")):
    #                         result[k] = fake_date.strftime("%Y-%m-%d")   # ISO
    #                     else:
    #                         result[k] = fake_date.strftime("%d-%m-%Y")
    #                 else:
    #                     result[k] = fake_date.strftime("%d-%m-%Y")

    #             # ── address ───────────────────────────────────────────────────
    #             elif "line_1" in k_lower:
    #                 result[k] = f"UNIT {self._faker_factory.faker.random_int(min=1, max=99)}"

    #             elif any(w in k_lower for w in ["line_2", "street"]):
    #                 result[k] = self._faker_factory.faker.street_address().upper()

    #             elif any(w in k_lower for w in ["suburb", "city", "town", "place_of_birth", "authority"]):
    #                 result[k] = self._faker_factory.faker.city().upper()

    #             elif any(w in k_lower for w in ["postcode", "zip"]):
    #                 result[k] = str(self._faker_factory.faker.postcode())

    #             # ── licence / card / barcode ──────────────────────────────────
    #             elif any(w in k_lower for w in ["licence_number", "license_number"]):
    #                 result[k] = str(self._faker_factory.faker.random_number(digits=9, fix_len=True))

    #             elif "card_number" in k_lower:
    #                 result[k] = f"D{self._faker_factory.faker.random_number(digits=7, fix_len=True)}"

    #             elif "barcode" in k_lower:
    #                 result[k] = f"ABnoteNZ{self._faker_factory.faker.random_number(digits=10, fix_len=True)}"

    #             # ── conditions ────────────────────────────────────────────────
    #             elif k_lower == "conditions":
    #                 import random
    #                 valid = ["S", "B", "E", "A", "V", "X", "Y", "Z"]
    #                 if random.random() < 0.30:
    #                     result[k] = ""
    #                 else:
    #                     codes = random.sample(valid, random.randint(1, 3))
    #                     result[k] = "".join(sorted(codes, key=valid.index))

    #             else:
    #                 result[k] = v

    #         # ── full_name ─────────────────────────────────────────────
    #         full_name_keys = [k for k in result if k.lower() == "full_name"]
    #         if full_name_keys:
    #             first  = next((v for k, v in result.items()
    #                         if any(w in k.lower() for w in ["first_name", "given_name"]) and v), "")
    #             middle = next((v for k, v in result.items() if "middle" in k.lower() and v), "")
    #             last   = next((v for k, v in result.items()
    #                         if any(w in k.lower() for w in ["last_name", "surname", "family_name"]) and v), "")
    #             name_parts = [p for p in [first, middle, last] if p]
    #             if name_parts:
    #                 for fn_key in full_name_keys:
    #                     result[fn_key] = " ".join(name_parts)

    #         return result

    #     elif isinstance(data, list):
    #         return [self._parse_custom_json(i, _shared) for i in data]

    #     return data

    def _parse_custom_json(self, data: Any, shared: dict | None = None) -> Any:
            """
            Recursive JSON faker with shared values across nested objects.

            Shared fields:
            - dob
            - expiry
            - names
            - sex
            - document number
            """

            import random
            import string

            if shared is None:
                shared = {}

            if isinstance(data, list):
                return [self._parse_custom_json(i, shared) for i in data]

            if not isinstance(data, dict):
                return data

            faker = self._faker_factory.faker

            if "dob_obj" not in shared:
                dob = faker.date_of_birth(minimum_age=18, maximum_age=70)

                shared.update({
                    "dob_obj": dob,
                    "dob_dd": dob.strftime("%d"),
                    "dob_mm": dob.strftime("%m"),
                    "dob_yy": dob.strftime("%y"),
                    "dob_yyyy": dob.strftime("%Y"),
                })

            if "expiry_obj" not in shared:
                shared["expiry_obj"] = faker.date_between(
                    start_date="+1y",
                    end_date="+5y",
                )

            shared.setdefault(
                "first_name",
                faker.first_name().upper(),
            )

            shared.setdefault(
                "last_name",
                faker.last_name().upper(),
            )

            shared.setdefault(
                "sex",
                random.choice(["M", "F"]),
            )

            shared.setdefault(
                "doc_number",
                "".join(random.choices(string.ascii_uppercase, k=2))
                + "".join(random.choices(string.digits, k=7))
            )

            dd = shared["dob_dd"]
            mm = shared["dob_mm"]
            yy = shared["dob_yy"]
            yyyy = shared["dob_yyyy"]

            expiry = shared["expiry_obj"]

            result = {}

            for key, value in data.items():

                kl = key.lower()

                if kl in {
                    "conditions_legend",
                    "nationality",
                    "issuing_country",
                    "document_type",
                    "licence_type",
                    "vicroads_notice",
                }:
                    result[key] = value
                    continue

                if isinstance(value, dict):
                    result[key] = self._parse_custom_json(value, shared)
                    continue

                if isinstance(value, list):

                    if kl in {"members", "cardholders"}:

                        members = []

                        for i in range(1, random.randint(1, 5) + 1):

                            first = faker.first_name().upper()
                            last = faker.last_name().upper()

                            middle = (
                                faker.first_name()[0].upper()
                                if random.random() > 0.4
                                else None
                            )

                            full_name = " ".join(
                                filter(None, [str(i), first, middle, last])
                            )

                            members.append({
                                "position": i,
                                "first_name": first,
                                "middle_initial": middle,
                                "last_name": last,
                                "full_name": full_name,
                            })

                        result[key] = members

                    else:
                        result[key] = self._parse_custom_json(value, shared)

                    continue

                # Names
                if any(x in kl for x in (
                    "first_name",
                    "given_name",
                    "given_names",
                )):
                    result[key] = shared["first_name"]
                    continue

                if any(x in kl for x in (
                    "last_name",
                    "surname",
                    "family_name",
                )):
                    result[key] = shared["last_name"]
                    continue

                if "middle" in kl:
                    result[key] = (
                        faker.first_name()[0].upper()
                        if value else None
                    )
                    continue

                if kl == "full_name":
                    result[key] = (
                        f"{shared['first_name']} "
                        f"{shared['last_name']}"
                    )
                    continue

                # DOB
                if (
                    any(x in kl for x in (
                        "dob",
                        "date_of_birth",
                        "birth",
                    ))
                    and "watermark" not in kl
                    and "place" not in kl
                ):

                    vs = str(value).strip()

                    if vs.isdigit():
                        result[key] = f"{dd}{mm}{yyyy}"

                    elif vs.startswith(("19", "20")) and "-" in vs:
                        result[key] = f"{yyyy}-{mm}-{dd}"

                    else:
                        result[key] = f"{dd}-{mm}-{yyyy}"

                    continue

                if "watermark" in kl:
                    result[key] = f"{dd}{mm}{yyyy}"
                    continue

                if "age_indicator" in kl:
                    result[key] = f"{mm}-{yy}"
                    continue

                # Expiry
                if any(x in kl for x in ("expiry", "expiration")):

                    vs = str(value).strip()

                    if "/" in vs and vs.count("/") == 1:
                        result[key] = expiry.strftime("%m/%Y")

                    elif "/" in vs:
                        result[key] = expiry.strftime("%d/%m/%Y")

                    elif vs.startswith(("19", "20")):
                        result[key] = expiry.strftime("%Y-%m-%d")

                    else:
                        result[key] = expiry.strftime("%d-%m-%Y")

                    continue

                # Sex / document
                if kl == "sex":
                    result[key] = shared["sex"]
                    continue

                if kl == "document_number":
                    result[key] = shared["doc_number"]
                    continue

                
                # MRZ
                if kl in {"mrz_line1", "mrx_line1"}:

                    family = shared.get("family_name", shared.get("last_name", "")).replace(" ", "<").upper()
                    given  = shared.get("given_names", shared.get("given_name", shared.get("first_name", ""))).replace(" ", "<").upper()

                    name = f"{family}<<{given}"

                    result[key] = f"P<AUS{name}".ljust(44, "<")[:44]  # ljust TRUOC roi cat
                    
                    continue

                if kl in {"mrz_line2", "mrx_line2"}:

                    def calc_chk(s: str) -> str:
                        w = [7, 3, 1]
                        return str(sum(
                            (0 if c == '<' else int(c) if c.isdigit() else ord(c.upper()) - 55) * w[i % 3]
                            for i, c in enumerate(s)
                        ) % 10)

                    dob6 = f"{yy}{mm}{dd}"
                    exp6 = expiry.strftime("%y%m%d")

                    doc9         = shared["doc_number"].ljust(9, "<")[:9]
                    personal_num = "<" * 16
                    personal_chk = "0"

                    comp_str = f"{doc9}<{dob6}{exp6}{personal_num}{personal_chk}"
                    comp_chk = calc_chk(comp_str)

                    line2 = f"{doc9}<AUS{dob6}{shared['sex']}{exp6}{personal_num}{personal_chk}{comp_chk}"

                    result[key] = line2[:44].ljust(44, "<")  # cat truoc, pad sau de khong mat comp_chk

                    continue
                # Numbers
                if any(x in kl for x in (
                    "licence_number",
                    "license_number",
                )):
                    result[key] = str(
                        faker.random_number(
                            digits=9,
                            fix_len=True,
                        )
                    )
                    continue

                if kl == "medicare_card_number":

                    p1 = faker.random_number(
                        digits=4,
                        fix_len=True,
                    )

                    p2 = faker.random_number(
                        digits=5,
                        fix_len=True,
                    )

                    p3 = faker.random_number(digits=1)

                    result[key] = f"{p1} {p2} {p3}"
                    continue

                if "card_number" in kl:
                    result[key] = (
                        f"D"
                        f"{faker.random_number(digits=7, fix_len=True)}"
                    )
                    continue

                if "barcode" in kl:
                    result[key] = (
                        f"ABnoteNZ"
                        f"{faker.random_number(digits=10, fix_len=True)}"
                    )
                    continue

                # Address
                if "line_1" in kl:
                    result[key] = (
                        f"UNIT "
                        f"{faker.random_int(min=1, max=99)}"
                    )
                    continue

                if any(x in kl for x in ("line_2", "street")):
                    result[key] = faker.street_address().upper()
                    continue

                if any(x in kl for x in (
                    "suburb",
                    "city",
                    "town",
                    "place_of_birth",
                    "authority",
                )):
                    result[key] = faker.city().upper()
                    continue

                if any(x in kl for x in ("postcode", "zip")):
                    result[key] = str(faker.postcode())
                    continue

                # Conditions
                if kl == "conditions":

                    valid = ["S", "B", "E", "A", "V", "X", "Y", "Z"]

                    if random.random() < 0.3:
                        result[key] = ""

                    else:
                        result[key] = " ".join(
                            sorted(
                                random.sample(
                                    valid,
                                    random.randint(1, 3),
                                ),
                                key=valid.index,
                            )
                        )

                    continue

                result[key] = value

            return result

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