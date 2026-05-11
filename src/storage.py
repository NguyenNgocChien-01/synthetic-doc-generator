"""
Module quản lý lưu trữ dữ liệu đầu ra.
Chịu trách nhiệm tạo cấu trúc thư mục, đặt tên tệp và
lưu {id}.jpg vào thư mục images/{doc_type} và {id}.json
vào thư mục labels/{doc_type} cho mỗi mẫu dữ liệu.
"""

import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Lỗi liên quan đến lưu trữ tệp."""
    pass


class StorageManager:
    """
    Quản lý lưu trữ các mẫu dữ liệu tổng hợp.

    Đảm bảo:
        - Mỗi mẫu có một cặp tệp duy nhất: {id}.jpg và {id}.json.
        - Cấu trúc thư mục: 
            - dataset/document/{loai_tai_lieu}/
            - dataset/labels/{loai_tai_lieu}/
        - Ghi log đầy đủ cho mỗi thao tác lưu trữ.
        - Xử lý lỗi an toàn không làm gián đoạn toàn bộ quá trình.
    """

    def __init__(
        self,
        dataset_dir: str,
        use_uuid: bool = True,
        id_prefix: str = "",
        id_zero_padding: int = 6,
        image_format: str = "JPEG",
        image_quality: int = 92,
    ):
        """
        Khởi tạo StorageManager.

        Tham số:
            dataset_dir: Thư mục gốc chứa tập dữ liệu đầu ra.
            use_uuid: Dùng UUID làm tên tệp thay vì số thứ tự.
            id_prefix: Tiền tố cho tên tệp (ví dụ: 'passport_').
            id_zero_padding: Số chữ số đệm 0 khi dùng số thứ tự.
            image_format: Định dạng ảnh đầu ra ('JPEG' hoặc 'PNG').
            image_quality: Chất lượng ảnh JPEG (1-95).
        """
        self.dataset_dir = Path(dataset_dir)
        self.use_uuid = use_uuid
        self.id_prefix = id_prefix
        self.id_zero_padding = id_zero_padding
        self.image_format = image_format.upper()
        self.image_quality = image_quality
        
        # Biến đếm thứ tự file theo ngày
        self._daily_counters: Dict[str, int] = {}

        # Bộ đếm cho từng loại tài liệu (tổng hợp chung)
        self._counters: Dict[str, int] = {}

        # Thống kê
        self._total_saved = 0
        self._total_errors = 0
        self._total_bytes_written = 0

        logger.info(
            "Khoi tao StorageManager, thu muc dau ra: %s, dinh dang anh: %s.",
            self.dataset_dir,
            self.image_format,
        )

    def ensure_directories(self, doc_type: str) -> Tuple[Path, Path]:
        """
        Tạo cấu trúc thư mục đầu ra cho hình ảnh và nhãn.

        Tham số:
            doc_type: Loại tài liệu.

        Trả về:
            Tuple chứa (đường_dẫn_thư_mục_ảnh, đường_dẫn_thư_mục_nhãn).
        """
        image_dir = self.dataset_dir / "documents" / doc_type
        label_dir = self.dataset_dir / "labels" / doc_type
        
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        return image_dir, label_dir

    def get_next_id(self, doc_type: str) -> str:
        """
        Lấy ID tiếp theo cho một mẫu dữ liệu.
        """
        if self.use_uuid:
            unique_id = str(uuid.uuid4()).replace("-", "")[:16]
            return f"{self.id_prefix}{unique_id}"

        if doc_type not in self._counters:
            self._counters[doc_type] = self._scan_existing_count(doc_type)

        self._counters[doc_type] += 1
        counter = self._counters[doc_type]
        return f"{self.id_prefix}{counter:0{self.id_zero_padding}d}"

    def save_sample(
        self,
        doc_type: str,
        sample_id: str,
        image: Image.Image,
        fields_data: Dict[str, Any],
        bounding_boxes: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """
        Lưu mẫu dữ liệu vào đúng thư mục phân loại.
        """
        image_dir, label_dir = self.ensure_directories(doc_type)

        ext = "jpg" if self.image_format == "JPEG" else "png"
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Số thứ tự theo ngày
        counter_key = f"{doc_type}_{date_str}"
        self._daily_counters[counter_key] = self._daily_counters.get(counter_key, 0) + 1
        
        # Tìm số thứ tự tiếp theo dựa trên thư mục nhãn (để giữ file đồng bộ)
        existing = list(label_dir.glob(f"{doc_type}_*.json"))
        if existing:
            last_stt = max(
                int(f.stem.split("_")[-1])
                for f in existing
                if f.stem.split("_")[-1].isdigit()
            )
        else:
            last_stt = 0
            
        stt = last_stt + 1
        
        filename = f"{doc_type}_{stt:05d}"
        
        # Gán đường dẫn lưu riêng biệt
        image_path = image_dir / f"{filename}.{ext}"
        json_path = label_dir / f"{filename}.json"

        # 1. Lưu ảnh
        try:
            image_bytes = self._encode_image(image)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_size = len(image_bytes)
        except (IOError, OSError) as loi:
            self._total_errors += 1
            raise StorageError(f"Khong the luu anh '{image_path}': {loi}") from loi

        # 2. Lưu JSON
        try:
            json_data = self._build_json_record(
                sample_id=sample_id,
                doc_type=doc_type,
                fields_data=fields_data,
                bounding_boxes=bounding_boxes or [],
                metadata=metadata or {},
                image_path=str(image_path),
                image_size_bytes=image_size,
            )
            json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
            json_bytes = json_str.encode("utf-8")

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_str)

        except (IOError, OSError) as loi:
            # Rollback nếu lưu JSON thất bại
            image_path.unlink(missing_ok=True)
            self._total_errors += 1
            raise StorageError(f"Khong the luu JSON '{json_path}': {loi}") from loi

        # 3. Cập nhật thống kê
        self._total_saved += 1
        self._total_bytes_written += image_size + len(json_bytes)

        logger.debug(
            "Da luu mau '%s' (%s): anh %d bytes.", sample_id, doc_type, image_size
        )
        return image_path, json_path

    def _encode_image(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()

        if self.image_format == "JPEG":
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(
                buffer,
                format="JPEG",
                quality=self.image_quality,
                optimize=True,
                progressive=True,
            )
        else:
            image.save(buffer, format="PNG", optimize=True)

        return buffer.getvalue()

    def _build_json_record(
        self,
        sample_id: str,
        doc_type: str,
        fields_data: Dict,
        bounding_boxes: List[Dict],
        metadata: Dict,
        image_path: str,
        image_size_bytes: int,
    ) -> Dict:
        # Lấy dữ liệu thực từ fields_data
        actual_data = fields_data.get("extracted_data", fields_data)

        # Danh sách các trường cần trích xuất theo cấu trúc phẳng
        target_fields = [
            "issuing_country", "document_number", "family_name", 
            "given_names", "nationality", "date_of_birth", 
            "sex", "date_of_issue", "date_of_expiry", 
            "place_of_birth", "authority"
        ]

        # Khởi tạo bản ghi với document_type
        record = {
            "document_type": fields_data.get("document_type", actual_data.get("document_type", doc_type))
        }

        # Đổ dữ liệu các trường vào cấu trúc phẳng
        for field in target_fields:
            record[field] = actual_data.get(field)
        
        # Xử lý MRZ
        record["mrx_line1"] = actual_data.get("mrx_line1", actual_data.get("mrz_line1"))
        record["mrz_line2"] = actual_data.get("mrz_line2")

        return record
    
    def _scan_existing_count(self, doc_type: str) -> int:
        """
        Đếm mẫu hiện có để tiếp tục sinh ID số.
        Quét thư mục labels thay vì thư mục chung.
        """
        label_dir = self.dataset_dir / "labels" / doc_type
        if not label_dir.exists():
            return 0
        count = len(list(label_dir.glob("*.json")))
        if count > 0:
            logger.info("Tim thay %d mau hien co trong labels/%s, tiep tuc tu so %d.", count, doc_type, count)
        return count

    def get_statistics(self) -> Dict:
        return {
            "tong_mau_da_luu": self._total_saved,
            "tong_loi_luu_tru": self._total_errors,
            "tong_dung_luong_bytes": self._total_bytes_written,
            "tong_dung_luong_mb": round(self._total_bytes_written / (1024 * 1024), 2),
            "bo_dem_theo_loai": dict(self._counters),
        }

    def get_dataset_summary(self, doc_type: Optional[str] = None) -> Dict:
        summary = {}

        images_base = self.dataset_dir / "images"
        labels_base = self.dataset_dir / "labels"

        if doc_type:
            doc_types = [doc_type]
        else:
            doc_types = [
                d.name for d in labels_base.iterdir() if d.is_dir()
            ] if labels_base.exists() else []

        for dt in doc_types:
            img_dir = images_base / dt
            lbl_dir = labels_base / dt
            
            if not lbl_dir.exists() and not img_dir.exists():
                summary[dt] = {"so_luong": 0, "dung_luong_mb": 0}
                continue

            jpg_files = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
            png_files = list(img_dir.glob("*.png")) if img_dir.exists() else []
            json_files = list(lbl_dir.glob("*.json")) if lbl_dir.exists() else []

            total_bytes = sum(f.stat().st_size for f in jpg_files + png_files + json_files)
            summary[dt] = {
                "so_luong": len(jpg_files) + len(png_files),
                "so_tep_json": len(json_files),
                "dung_luong_mb": round(total_bytes / (1024 * 1024), 2),
            }

        return summary