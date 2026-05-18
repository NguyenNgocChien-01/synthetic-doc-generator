"""
Module xử lý ảnh: render văn bản lên template và áp dụng biến đổi ngẫu nhiên.
Sử dụng Pillow để vẽ văn bản, dán ảnh đại diện và tạo hiệu ứng thực tế.
"""

import io
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

logger = logging.getLogger(__name__)

# Danh sách font hệ thống theo thứ tự ưu tiên
SYSTEM_FONTS_PRIORITY = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]

SYSTEM_FONTS_BOLD_PRIORITY = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/HelveticaBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

SYSTEM_FONTS_MONO_PRIORITY = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Courier New.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def _load_best_font(
    font_paths: List[str],
    size: int,
    fallback_size: Optional[int] = None,
) -> ImageFont.FreeTypeFont:
    """
    Tải font tốt nhất có sẵn từ danh sách ưu tiên.

    Tham số:
        font_paths: Danh sách đường dẫn font theo thứ tự ưu tiên.
        size: Kích thước font.
        fallback_size: Kích thước dự phòng (dùng font mặc định).

    Trả về:
        Đối tượng ImageFont.
    """ 
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue

    # Dự phòng: dùng font mặc định của PIL
    logger.debug("Khong tim thay font he thong, dung font mac dinh cua PIL.")
    return ImageFont.load_default()


class FontCache:
    """Bộ nhớ đệm font để tránh tải lại nhiều lần."""

    def __init__(self):
        self._cache: Dict[Tuple, ImageFont.FreeTypeFont] = {}

    def get(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """
        Lấy font từ bộ nhớ đệm hoặc tải mới.

        Tham số:
            style: Kiểu font ('regular', 'bold', 'mono').
            size: Kích thước font.

        Trả về:
            Đối tượng ImageFont.
        """
        key = (style, size)
        if key not in self._cache:
            if style == "bold":
                font_paths = SYSTEM_FONTS_BOLD_PRIORITY
            elif style == "mono":
                font_paths = SYSTEM_FONTS_MONO_PRIORITY
            else:
                font_paths = SYSTEM_FONTS_PRIORITY
            self._cache[key] = _load_best_font(font_paths, size)
        return self._cache[key]


class ImageProcessor:
    """
    Xử lý ảnh tài liệu: render văn bản, dán ảnh và áp dụng biến đổi.

    Chức năng chính:
        - Tải ảnh template hoặc tạo nền tài liệu mới.
        - Vẽ các trường văn bản tại vị trí được định nghĩa trong base.json.
        - Dán ảnh đại diện vào vị trí được chỉ định.
        - Áp dụng các biến đổi ngẫu nhiên (rotation, noise, brightness) để
          tăng tính đa dạng cho tập huấn luyện OCR.
    """

    def __init__(self, templates_dir: str, config: Any = None):
        """
        Khởi tạo ImageProcessor.

        Tham số:
            templates_dir: Đường dẫn thư mục chứa các template.
            config: Đối tượng cấu hình (ImageConfig).
        """
        self.templates_dir = Path(templates_dir)
        self.config = config
        self.font_cache = FontCache()
        self._template_cache: Dict[str, Image.Image] = {}

        logger.debug(
            "Khoi tao ImageProcessor, thu muc template: %s.", self.templates_dir
        )

    def render_document(
        self,
        doc_type: str,
        template_config: Dict,
        fields_data: Dict[str, str],
        avatar_image: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Render tài liệu đầy đủ với dữ liệu đã cung cấp.

        Tham số:
            doc_type: Loại tài liệu (passport, driver_license, ...).
            template_config: Cấu hình từ base.json.
            fields_data: Dữ liệu các trường đã được sinh.
            avatar_image: Ảnh đại diện (tùy chọn).

        Trả về:
            Tuple (anh_da_render, danh_sach_bounding_box).
        """
        # Tải ảnh template
        image = self._load_template_image(doc_type, template_config)
        image = image.copy()  # Tạo bản sao để không ảnh hưởng đến cache

        draw = ImageDraw.Draw(image)
        bounding_boxes = []

        # Dán ảnh đại diện nếu có
        avatar_config = template_config.get("avatar", {})
        if avatar_config.get("enabled", False) and avatar_image:
            bbox = self._paste_avatar(image, avatar_image, avatar_config)
            if bbox:
                bounding_boxes.append({
                    "key": "avatar",
                    "type": "image",
                    "bounding_box": bbox,
                })

        # Vẽ từng trường văn bản
        for field_cfg in template_config.get("fields", []):
            key = field_cfg.get("key", "")
            value = fields_data.get(key, "")
            if not value:
                continue

            bbox = self._draw_field(draw, image, field_cfg, value)
            if bbox:
                bounding_boxes.append({
                    "key": key,
                    "value": value,
                    "bounding_box": bbox,
                    "faker_type": field_cfg.get("faker_type", ""),
                })

        # Áp dụng biến đổi ngẫu nhiên nếu được bật
        should_augment = True
        if self.config and not self.config.enable_augmentation:
            should_augment = False

        if should_augment:
            image = self._apply_augmentation(image)

        logger.debug(
            "Da render tai lieu '%s' voi %d truong va %d bounding box.",
            doc_type,
            len(fields_data),
            len(bounding_boxes),
        )
        return image, bounding_boxes
    
    def _load_template_image(self, doc_type: str, template_config: Dict, state: str = None) -> Image.Image:
        cache_key = f"{doc_type}_{state}" if state else doc_type
        if cache_key in self._template_cache:
            return self._template_cache[cache_key].copy()
    
        if state:
            template_dir = self.templates_dir / doc_type / state
        else:
            template_dir = self.templates_dir / doc_type
    
        template_path = None
        for ext in ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]:
            for candidate in template_dir.glob(f"template.{ext}"):
                template_path = candidate
                break
            if template_path:
                break
    
        if template_path and template_path.exists():
            try:
                img = Image.open(template_path).convert("RGB")
                self._template_cache[cache_key] = img
                logger.debug("Da tai anh template tu: %s.", template_path)
                return img.copy()
            except Exception as loi:
                logger.warning("Khong the tai template '%s': %s.", template_path, loi)
    
        img_size = template_config.get("image_size", [1200, 850])
        img = self._generate_document_background(doc_type, img_size, template_config)
        self._template_cache[cache_key] = img
        return img.copy()
            

    def _generate_document_background(
        self,
        doc_type: str,
        size: List[int],
        template_config: Dict,
    ) -> Image.Image:
        """
        Tạo ảnh nền tài liệu giả đơn giản khi không có template thực.

        Tham số:
            doc_type: Loại tài liệu.
            size: [chiều rộng, chiều cao].
            template_config: Cấu hình template.

        Trả về:
            Đối tượng PIL Image đã vẽ nền.
        """
        width, height = size[0], size[1]

        # Màu nền theo loại tài liệu
        bg_colors = {
            "passport": (245, 245, 240),
            "driver_license": (235, 240, 250),
            "medicare_card": (230, 245, 240),
            "utility_bill": (250, 250, 245),
        }
        bg_color = bg_colors.get(doc_type, (245, 245, 245))
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Vẽ viền
        border_colors = {
            "passport": (0, 60, 120),
            "driver_license": (20, 80, 160),
            "medicare_card": (0, 100, 80),
            "utility_bill": (80, 60, 20),
        }
        border_color = border_colors.get(doc_type, (0, 80, 160))

        draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=8)
        draw.rectangle([10, 10, width - 11, height - 11], outline=border_color, width=2)

        # Vẽ vùng tiêu đề
        header_height = 80
        draw.rectangle([0, 0, width, header_height], fill=border_color)

        # Tiêu đề tài liệu
        titles = {
            "passport": "PASSPORT / HO CHIEU",
            "driver_license": "DRIVER LICENCE",
            "medicare_card": "MEDICARE CARD",
            "utility_bill": "UTILITY BILL / HOA DON TIEN ICH",
        }
        title_text = titles.get(doc_type, doc_type.upper().replace("_", " "))

        font_title = self.font_cache.get("bold", 32)
        draw.text(
            (width // 2, header_height // 2),
            title_text,
            fill=(255, 255, 255),
            font=font_title,
            anchor="mm",
        )

        # Vẽ các đường phân chia trường dữ liệu
        field_line_color = (200, 200, 200)
        for field_cfg in template_config.get("fields", []):
            pos = field_cfg.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
            max_w = field_cfg.get("max_width", 300)
            fs = field_cfg.get("font_size", 18)

            # Vẽ nhãn trường
            label = field_cfg.get("label", field_cfg.get("key", ""))
            font_label = self.font_cache.get("regular", max(10, fs - 6))
            draw.text((x, y - (fs - 2)), label.upper(), fill=(120, 120, 120), font=font_label)

            # Vẽ đường gạch chân
            draw.line([(x, y + fs + 4), (x + max_w, y + fs + 4)], fill=field_line_color, width=1)

        return img

    def render_document_on_base(
        self,
        base_image: Image.Image,
        doc_type: str,
        template_config: Dict,
        fields_data: Dict[str, str],
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Thực thi Giai đoạn 2: Vẽ văn bản tĩnh lên ảnh nền đồ họa từ API.
        """
        image = base_image.copy()
        draw = ImageDraw.Draw(image)
        bounding_boxes = []

        # Chỉ lặp qua cấu hình JSON để vẽ văn bản
        for field_cfg in template_config.get("fields", []):
            key = field_cfg.get("key", "")
            value = fields_data.get(key, "")
            if not value:
                continue

            bbox = self._draw_field(draw, image, field_cfg, value)
            if bbox:
                bounding_boxes.append({
                    "key": key,
                    "value": value,
                    "bounding_box": bbox,
                    "faker_type": field_cfg.get("faker_type", ""),
                })

        # Áp dụng Augmentation sau khi đã có toàn bộ văn bản
        should_augment = True
        if self.config and not self.config.enable_augmentation:
            should_augment = False

        if should_augment:
            image = self._apply_augmentation(image)

        return image, bounding_boxes
    def _draw_field(
        self,
        draw: ImageDraw.Draw,
        image: Image.Image,
        field_cfg: Dict,
        value: str,
    ) -> Optional[List[int]]:
        """
        Vẽ một trường văn bản lên ảnh tại vị trí xác định.

        Tham số:
            draw: Đối tượng ImageDraw.
            image: Ảnh đang vẽ lên.
            field_cfg: Cấu hình trường từ base.json.
            value: Giá trị văn bản cần vẽ.

        Trả về:
            Bounding box [x1, y1, x2, y2] của văn bản đã vẽ, hoặc None nếu thất bại.
        """
        pos = field_cfg.get("position", {})
        x = pos.get("x", 0)
        y = pos.get("y", 0)

        font_size = field_cfg.get("font_size", 20)
        font_style = field_cfg.get("font_style", "regular")
        font_color = tuple(field_cfg.get("font_color", [0, 0, 0]))
        max_width = field_cfg.get("max_width", None)
        anchor = field_cfg.get("anchor", "lt")  # lt = left-top
        rotation = field_cfg.get("rotation", 0)

        # Lấy font từ cache
        font = self.font_cache.get(font_style, font_size)

        try:
            if rotation != 0:
                # Vẽ văn bản xoay (dùng cho một số trường đặc biệt)
                bbox = self._draw_rotated_text(
                    image, x, y, value, font, font_color, rotation
                )
            else:
                # Vẽ văn bản thẳng
                bbox = draw.textbbox((x, y), value, font=font, anchor=anchor)
                draw.text((x, y), value, fill=font_color, font=font, anchor=anchor)

            return [int(b) for b in bbox]

        except Exception as loi:
            logger.warning(
                "Loi khi ve truong '%s' (gia tri='%s'): %s",
                field_cfg.get("key", "?"),
                value[:20],
                loi,
            )
            return None

    def _draw_rotated_text(
        self,
        image: Image.Image,
        x: int,
        y: int,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: tuple,
        angle: float,
    ) -> List[int]:
        """
        Vẽ văn bản xoay góc bằng cách tạo layer trung gian.

        Tham số:
            image: Ảnh chính cần vẽ lên.
            x, y: Tọa độ bắt đầu.
            text: Nội dung văn bản.
            font: Font chữ.
            color: Màu chữ.
            angle: Góc xoay (độ).

        Trả về:
            Bounding box xấp xỉ [x1, y1, x2, y2].
        """
        # Tạo ảnh tạm cho văn bản
        tmp_draw = ImageDraw.Draw(image)
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        tmp_img = Image.new("RGBA", (text_width + 20, text_height + 20), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp_img)
        tmp_draw.text((10, 10), text, fill=(*color, 255), font=font)

        # Xoay ảnh tạm
        rotated = tmp_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        # Dán vào ảnh chính
        image.paste(rotated, (x, y), rotated)

        return [x, y, x + rotated.width, y + rotated.height]

    def _paste_avatar(
        self,
        image: Image.Image,
        avatar: Image.Image,
        avatar_config: Dict,
    ) -> Optional[List[int]]:
        """
        Dán ảnh đại diện vào vị trí xác định trên tài liệu.

        Tham số:
            image: Ảnh tài liệu chính.
            avatar: Ảnh đại diện cần dán.
            avatar_config: Cấu hình vị trí và kích thước.

        Trả về:
            Bounding box [x1, y1, x2, y2] hoặc None.
        """
        pos = avatar_config.get("position", {"x": 50, "y": 100})
        size = avatar_config.get("size", [150, 190])

        x, y = pos.get("x", 50), pos.get("y", 100)
        w, h = size[0], size[1]

        try:
            avatar_resized = avatar.resize((w, h), Image.Resampling.LANCZOS)

            # Vẽ viền cho ảnh đại diện
            draw = ImageDraw.Draw(image)
            border_width = 2
            draw.rectangle(
                [x - border_width, y - border_width, x + w + border_width, y + h + border_width],
                outline=(100, 100, 100),
                width=border_width,
            )

            image.paste(avatar_resized, (x, y))
            return [x, y, x + w, y + h]

        except Exception as loi:
            logger.warning("Loi khi dan anh dai dien: %s", loi)
            return None

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Áp dụng các biến đổi ngẫu nhiên để tăng tính đa dạng.

        Biến đổi bao gồm: xoay nhỏ, điều chỉnh độ sáng,
        thêm nhiễu Gaussian và làm mờ nhẹ.

        Tham số:
            image: Ảnh gốc cần biến đổi.

        Trả về:
            Ảnh đã được biến đổi.
        """
        # 1. Xoay nhỏ (mô phỏng ảnh chụp hơi nghiêng)
        max_rotation = 2.0
        if self.config:
            max_rotation = self.config.augmentation_rotation_max_degrees

        if max_rotation > 0:
            angle = random.uniform(-max_rotation, max_rotation)
            if abs(angle) > 0.3:
                image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

        # 2. Điều chỉnh độ sáng
        brightness_range = (0.90, 1.10)
        if self.config:
            brightness_range = self.config.augmentation_brightness_range

        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # 3. Điều chỉnh độ tương phản nhẹ
        contrast_factor = random.uniform(0.92, 1.08)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # 4. Thêm nhiễu Gaussian
        noise_sigma = 3.0
        if self.config:
            noise_sigma = self.config.augmentation_noise_sigma

        if noise_sigma > 0:
            image = self._add_gaussian_noise(image, sigma=noise_sigma)

        # 5. Làm mờ nhẹ (xác suất thấp)
        blur_prob = 0.15
        blur_radius = 0.8
        if self.config:
            blur_prob = self.config.augmentation_blur_probability
            blur_radius = self.config.augmentation_blur_radius

        if random.random() < blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return image

    def _add_gaussian_noise(self, image: Image.Image, sigma: float = 3.0) -> Image.Image:
        """
        Thêm nhiễu Gaussian vào ảnh.

        Tham số:
            image: Ảnh gốc.
            sigma: Độ lệch chuẩn của nhiễu.

        Trả về:
            Ảnh đã thêm nhiễu.
        """
        try:
            import numpy as np
            img_array = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, sigma, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)
        except ImportError:
            # Không có numpy: bỏ qua bước thêm nhiễu
            return image
        except Exception as loi:
            logger.debug("Khong the them nhieu Gaussian: %s", loi)
            return image

    def clear_template_cache(self) -> None:
        """Xóa bộ nhớ đệm template để giải phóng bộ nhớ."""
        self._template_cache.clear()
        logger.debug("Da xoa bo nho dem template.")
