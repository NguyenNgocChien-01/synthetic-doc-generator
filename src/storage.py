"""
Module quản lý lưu trữ dữ liệu đầu ra.
Chịu trách nhiệm tạo cấu trúc thư mục, đặt tên tệp và
lưu cặp {id}.jpg + {id}.json cho mỗi mẫu dữ liệu.
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
        - Cấu trúc thư mục nhất quán: dataset/{loai_tai_lieu}/.
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

        # Bộ đếm cho từng loại tài liệu
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

    def ensure_directory(self, doc_type: str) -> Path:
        """
        Tạo thư mục đầu ra cho loại tài liệu nếu chưa tồn tại.

        Tham số:
            doc_type: Loại tài liệu.

        Trả về:
            Đường dẫn thư mục đã tạo.
        """
        output_dir = self.dataset_dir / doc_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_next_id(self, doc_type: str) -> str:
        """
        Lấy ID tiếp theo cho một mẫu dữ liệu.

        Nếu dùng UUID: trả về chuỗi UUID ngẫu nhiên.
        Nếu dùng số thứ tự: trả về số đệm 0 kết hợp tiền tố.

        Tham số:
            doc_type: Loại tài liệu.

        Trả về:
            Chuỗi ID duy nhất.
        """
        if self.use_uuid:
            unique_id = str(uuid.uuid4()).replace("-", "")[:16]
            return f"{self.id_prefix}{unique_id}"

        # Khởi tạo bộ đếm nếu chưa có
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
        fields_data: Dict[str, str],
        bounding_boxes: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """
        Lưu một mẫu dữ liệu: ảnh JPG và tệp JSON đi kèm.

        Cấu trúc tệp JSON:
        {
            "id": "...",
            "doc_type": "...",
            "created_at": "...",
            "fields": { "key": "value", ... },
            "bounding_boxes": [ { "key": "...", "bounding_box": [...] }, ... ],
            "metadata": { ... }
        }

        Tham số:
            doc_type: Loại tài liệu.
            sample_id: ID duy nhất của mẫu.
            image: Ảnh đã render.
            fields_data: Dữ liệu các trường văn bản.
            bounding_boxes: Danh sách bounding box OCR.
            metadata: Thông tin bổ sung tùy chọn.

        Trả về:
            Tuple (duong_dan_anh, duong_dan_json).

        Ném ra:
            StorageError: Nếu không thể lưu tệp.
        """
        output_dir = self.ensure_directory(doc_type)

        # Xác định tên tệp theo định dạng
        ext = "jpg" if self.image_format == "JPEG" else "png"
        image_path = output_dir / f"{sample_id}.{ext}"
        json_path = output_dir / f"{sample_id}.json"

        # Lưu ảnh
        try:
            image_bytes = self._encode_image(image)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_size = len(image_bytes)
        except (IOError, OSError) as loi:
            self._total_errors += 1
            raise StorageError(
                f"Khong the luu anh '{image_path}': {loi}"
            ) from loi

        # Lưu JSON
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
            # Xóa ảnh đã lưu để đảm bảo tính nhất quán
            image_path.unlink(missing_ok=True)
            self._total_errors += 1
            raise StorageError(
                f"Khong the luu JSON '{json_path}': {loi}"
            ) from loi

        # Cập nhật thống kê
        self._total_saved += 1
        self._total_bytes_written += image_size + len(json_bytes)

        logger.debug(
            "Da luu mau '%s' (%s): anh %d bytes.", sample_id, doc_type, image_size
        )
        return image_path, json_path

    def _encode_image(self, image: Image.Image) -> bytes:
        """
        Mã hóa ảnh thành bytes theo định dạng cấu hình.

        Tham số:
            image: Ảnh PIL cần mã hóa.

        Trả về:
            Mảng bytes ảnh đã nén.
        """
        buffer = io.BytesIO()

        if self.image_format == "JPEG":
            # Đảm bảo ảnh là RGB (JPEG không hỗ trợ RGBA)
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
        fields_data: Dict[str, str],
        bounding_boxes: List[Dict],
        metadata: Dict,
        image_path: str,
        image_size_bytes: int,
    ) -> Dict:
        """
        Xây dựng cấu trúc JSON chuẩn cho một mẫu dữ liệu OCR.

        Tham số:
            (xem save_sample)

        Trả về:
            Từ điển Python sẵn sàng để tuần tự hóa JSON.
        """
        # Chuyển bounding box sang định dạng chuẩn COCO-style
        annotations = []
        for bb in bounding_boxes:
            if not bb.get("bounding_box"):
                continue
            x1, y1, x2, y2 = bb["bounding_box"]
            annotations.append({
                "key": bb.get("key", ""),
                "value": bb.get("value", ""),
                "faker_type": bb.get("faker_type", ""),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            })

        return {
            "id": sample_id,
            "doc_type": doc_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "image_file": Path(image_path).name,
            "image_size_bytes": image_size_bytes,
            "fields": fields_data,
            "annotations": annotations,
            "metadata": {
                **metadata,
                "generator": "SyntheticDocGenerator",
                "version": "1.0",
            },
        }

    def _scan_existing_count(self, doc_type: str) -> int:
        """
        Đếm số mẫu hiện có trong thư mục để tiếp tục đánh số.

        Tham số:
            doc_type: Loại tài liệu.

        Trả về:
            Số mẫu hiện có.
        """
        output_dir = self.dataset_dir / doc_type
        if not output_dir.exists():
            return 0
        count = len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.png")))
        if count > 0:
            logger.info(
                "Tim thay %d mau hien co trong '%s', tiep tuc tu so %d.",
                count, doc_type, count,
            )
        return count

    def get_statistics(self) -> Dict:
        """
        Lấy thống kê tổng hợp về hoạt động lưu trữ.

        Trả về:
            Từ điển thống kê.
        """
        return {
            "tong_mau_da_luu": self._total_saved,
            "tong_loi_luu_tru": self._total_errors,
            "tong_dung_luong_bytes": self._total_bytes_written,
            "tong_dung_luong_mb": round(self._total_bytes_written / (1024 * 1024), 2),
            "bo_dem_theo_loai": dict(self._counters),
        }

    def get_dataset_summary(self, doc_type: Optional[str] = None) -> Dict:
        """
        Tóm tắt tập dữ liệu hiện có trên đĩa.

        Tham số:
            doc_type: Lọc theo loại tài liệu. None = tất cả.

        Trả về:
            Từ điển tóm tắt tập dữ liệu.
        """
        summary = {}

        if doc_type:
            doc_types = [doc_type]
        else:
            doc_types = [
                d.name for d in self.dataset_dir.iterdir()
                if d.is_dir()
            ] if self.dataset_dir.exists() else []

        for dt in doc_types:
            dir_path = self.dataset_dir / dt
            if not dir_path.exists():
                summary[dt] = {"so_luong": 0, "dung_luong_mb": 0}
                continue

            jpg_files = list(dir_path.glob("*.jpg"))
            png_files = list(dir_path.glob("*.png"))
            json_files = list(dir_path.glob("*.json"))

            total_bytes = sum(f.stat().st_size for f in jpg_files + png_files + json_files)
            summary[dt] = {
                "so_luong": len(jpg_files) + len(png_files),
                "so_tep_json": len(json_files),
                "dung_luong_mb": round(total_bytes / (1024 * 1024), 2),
            }

        return summary
