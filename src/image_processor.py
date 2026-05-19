"""
Module for image processing: rendering text on templates and applying random transformations.
Uses Pillow to draw text, paste avatars, and create realistic effects.
"""

import io
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

logger = logging.getLogger(__name__)

# System font priority list
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
    Load the best available font from the priority list.

    Args:
        font_paths: List of font paths in priority order.
        size: Font size.
        fallback_size: Fallback size (uses default font).

    Returns:
        ImageFont object.
    """ 
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue

    # Fallback: use PIL default font
    logger.debug("System fonts not found, using PIL default font.")
    return ImageFont.load_default()

class FontCache:
    """Font cache to avoid reloading fonts multiple times."""

    def __init__(self):
        self._cache: Dict[Tuple, ImageFont.FreeTypeFont] = {}

    def get(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """
        Retrieve font from cache or load a new one.

        Args:
            style: Font style ('regular', 'bold', 'mono').
            size: Font size.

        Returns:
            ImageFont object.
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
    Document image processing: render text, paste images, and apply transformations.

    Main functions:
        - Load template images or create new document backgrounds.
        - Draw text fields at positions defined in base.json.
        - Paste avatar images at specified positions.
        - Apply random transformations (rotation, noise, brightness) to
          increase diversity for OCR training datasets.
    """

    def __init__(self, templates_dir: str, config: Any = None):
        """
        Initialize ImageProcessor.

        Args:
            templates_dir: Path to the directory containing templates.
            config: Configuration object (ImageConfig).
        """
        self.templates_dir = Path(templates_dir)
        self.config = config
        self.font_cache = FontCache()
        self._template_cache: Dict[str, Image.Image] = {}

        logger.debug(
            "Initialized ImageProcessor, template directory: %s.", self.templates_dir
        )

    def render_document(
        self,
        doc_type: str,
        template_config: Dict,
        fields_data: Dict[str, str],
        avatar_image: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Render a complete document with the provided data.

        Args:
            doc_type: Document type (passport, driver_license, etc.).
            template_config: Configuration from base.json.
            fields_data: Generated field data.
            avatar_image: Optional avatar image.

        Returns:
            Tuple (rendered_image, bounding_boxes).
        """
        # Load template image
        image = self._load_template_image(doc_type, template_config)
        image = image.copy()  # Create a copy to avoid affecting the cache

        draw = ImageDraw.Draw(image)
        bounding_boxes = []

        # Paste avatar image if available
        avatar_config = template_config.get("avatar", {})
        if avatar_config.get("enabled", False) and avatar_image:
            bbox = self._paste_avatar(image, avatar_image, avatar_config)
            if bbox:
                bounding_boxes.append({
                    "key": "avatar",
                    "type": "image",
                    "bounding_box": bbox,
                })

        # Draw each text field
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

        # Apply random transformations if enabled
        should_augment = True
        if self.config and not self.config.enable_augmentation:
            should_augment = False

        if should_augment:
            image = self._apply_augmentation(image)

        logger.debug(
            "Rendered document '%s' with %d fields and %d bounding boxes.",
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
                logger.debug("Loaded template from: %s.", template_path)
                return img.copy()
            except Exception as loi:
                logger.warning("Could not load template '%s': %s.", template_path, loi)
    
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
        Create a simple background image for documents when no template is available.

        Args:
            doc_type: Document type.
            size: [width, height].
            template_config: Template configuration.

        Returns:
            PIL Image object.
        """
        width, height = size[0], size[1]

        # Background color based on document type
        bg_colors = {
            "passport": (245, 245, 240),
            "driver_license": (235, 240, 250),
            "medicare_card": (230, 245, 240),
            "utility_bill": (250, 250, 245),
        }
        bg_color = bg_colors.get(doc_type, (245, 245, 245))
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw border
        border_colors = {
            "passport": (0, 60, 120),
            "driver_license": (20, 80, 160),
            "medicare_card": (0, 100, 80),
            "utility_bill": (80, 60, 20),
        }
        border_color = border_colors.get(doc_type, (0, 80, 160))

        draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=8)
        draw.rectangle([10, 10, width - 11, height - 11], outline=border_color, width=2)

        # Draw header
        header_height = 80
        draw.rectangle([0, 0, width, header_height], fill=border_color)

        # Document title
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

        # Draw field lines
        field_line_color = (200, 200, 200)
        for field_cfg in template_config.get("fields", []):
            pos = field_cfg.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
            max_w = field_cfg.get("max_width", 300)
            fs = field_cfg.get("font_size", 18)

            # Draw field label
            label = field_cfg.get("label", field_cfg.get("key", ""))
            font_label = self.font_cache.get("regular", max(10, fs - 6))
            draw.text((x, y - (fs - 2)), label.upper(), fill=(120, 120, 120), font=font_label)

            # Draw field line
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
        Execute Phase 2: Draw static text on a base image from API.
        """
        image = base_image.copy()
        draw = ImageDraw.Draw(image)
        bounding_boxes = []

        # Only loop through JSON configuration to draw text
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

        # Apply Augmentation after having all text
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
        Draw a text field on an image at a specific position.

        Args:
            draw: ImageDraw object.
            image: Image to draw on.
            field_cfg: Field configuration from base.json.
            value: Text value to draw.

        Returns:
            Bounding box [x1, y1, x2, y2] of the drawn text, or None if failed.
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

        # Load font from cache
        font = self.font_cache.get(font_style, font_size)

        try:
            if rotation != 0:
                # Draw text rotated (for some special fields)
                bbox = self._draw_rotated_text(
                    image, x, y, value, font, font_color, rotation
                )
            else:
                # Draw text straight
                bbox = draw.textbbox((x, y), value, font=font, anchor=anchor)
                draw.text((x, y), value, fill=font_color, font=font, anchor=anchor)

            return [int(b) for b in bbox]

        except Exception as loi:
            logger.warning(
                "Error when drawing field '%s' (value='%s'): %s",
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
        Draw text rotated by an angle using an intermediate layer.

        Args:
            image: Main image to draw on.
            x, y: Starting position.
            text: Text content.
            font: Font.
            color: Color.
            angle: Angle (degrees).

        Returns:
            Bounding box [x1, y1, x2, y2].
        """
        # Create a temporary image for the text
        tmp_draw = ImageDraw.Draw(image)
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        tmp_img = Image.new("RGBA", (text_width + 20, text_height + 20), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp_img)
        tmp_draw.text((10, 10), text, fill=(*color, 255), font=font)

        # Rotate the temporary image
        rotated = tmp_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        # Paste into the main image
        image.paste(rotated, (x, y), rotated)

        return [x, y, x + rotated.width, y + rotated.height]

    def _paste_avatar(
        self,
        image: Image.Image,
        avatar: Image.Image,
        avatar_config: Dict,
    ) -> Optional[List[int]]:
        """
        Paste an avatar image into a specific position on the document.

        Args:
            image: Main image.
            avatar: Avatar image.
            avatar_config: Configuration for position and size.

        Returns:
            Bounding box [x1, y1, x2, y2] or None.
        """
        pos = avatar_config.get("position", {"x": 50, "y": 100})
        size = avatar_config.get("size", [150, 190])

        x, y = pos.get("x", 50), pos.get("y", 100)
        w, h = size[0], size[1]

        try:
            avatar_resized = avatar.resize((w, h), Image.Resampling.LANCZOS)

            # Draw border for the avatar
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
            logger.warning("Error when pasting avatar: %s", loi)
            return None

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply random transformations to increase diversity.

        Transformations include: small rotation, brightness adjustment,
        Gaussian noise, and light blurring.

        Args:
            image: Original image.

        Returns:
            Transformed image.
        """
        # 1. Small rotation (mimics slightly tilted images)
        max_rotation = 2.0
        if self.config:
            max_rotation = self.config.augmentation_rotation_max_degrees

        if max_rotation > 0:
            angle = random.uniform(-max_rotation, max_rotation)
            if abs(angle) > 0.3:
                image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

        # 2. Brightness adjustment
        brightness_range = (0.90, 1.10)
        if self.config:
            brightness_range = self.config.augmentation_brightness_range

        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # 3. Contrast adjustment
        contrast_factor = random.uniform(0.92, 1.08)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # 4. Gaussian noise
        noise_sigma = 3.0
        if self.config:
            noise_sigma = self.config.augmentation_noise_sigma

        if noise_sigma > 0:
            image = self._add_gaussian_noise(image, sigma=noise_sigma)

        # 5. Light blurring
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
        Add Gaussian noise to the image.

        Args:
            image: Original image.
            sigma: Standard deviation of the noise.

        Returns:
            Image with added noise.
        """
        try:
            import numpy as np
            img_array = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, sigma, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)
        except ImportError:
            # No numpy: skip the noise step
            return image
        except Exception as loi:
            logger.debug("Could not add Gaussian noise: %s", loi)
            return image

    def clear_template_cache(self) -> None:
        """Clear the template cache to free memory."""
        self._template_cache.clear()
        logger.debug("Template cache cleared.")


