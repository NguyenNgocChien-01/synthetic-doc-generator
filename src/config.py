# src/config.py
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

PLATFORM_VERTEX_AI = "vertex_ai"
SUPPORTED_PLATFORMS = [PLATFORM_VERTEX_AI]

SUPPORTED_DOC_TYPES = [
    "aus_passport",
    "driver_license",
    "aus_medicare_card",
    "utility_bill",
]

PLATFORM_ENDPOINTS: Dict[str, str] = {
    PLATFORM_VERTEX_AI: "https://{region}-aiplatform.googleapis.com/v1",
}

@dataclass
class APIConfig:
    platform: str = PLATFORM_VERTEX_AI
    endpoint: Optional[str] = None
    project_id: Optional[str] = None
    region: str = "us-central1"
    model_name: str = "gemini-1.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    retry_backoff_factor: float = 2.0

    def resolve_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint
        base = PLATFORM_ENDPOINTS.get(self.platform, "")
        return base.format(region=self.region)

@dataclass
class ImageConfig:
    output_format: str = "JPEG"
    output_quality: int = 92
    enable_augmentation: bool = True
    augmentation_rotation_max_degrees: float = 2.0
    augmentation_brightness_range: tuple = (0.90, 1.10)
    augmentation_noise_sigma: float = 3.0
    augmentation_blur_probability: float = 0.15
    augmentation_blur_radius: float = 0.8
    avatar_use_api: bool = False
    avatar_placeholder_service: str = "robohash"

@dataclass
class StorageConfig:
    templates_dir: str = "templates"
    dataset_dir: str = "dataset"
    log_dir: str = "logs"
    use_uuid_filenames: bool = True
    id_prefix: str = ""
    id_zero_padding: int = 6

@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    num_workers: int = 4
    batch_size: int = 10
    debug_mode: bool = False
    dry_run: bool = False

    @classmethod
    def from_cli_args(cls, args: Any) -> "Config":
        config = cls()
        
        config.api.platform = PLATFORM_VERTEX_AI
        config.api.endpoint = getattr(args, "endpoint", None)
        config.api.project_id = (
            getattr(args, "project_id", None)
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        config.api.region = getattr(args, "region", "us-central1")
        config.api.model_name = getattr(args, "model", "gemini-1.5-flash")
        config.api.image_model = getattr(args, "image_model", "gemini-2.5-flash-image")
        config.api.max_retries = getattr(args, "max_retries", 3)

        config.storage.templates_dir = getattr(args, "templates_dir", "templates")
        config.storage.dataset_dir = getattr(args, "output_dir", "dataset")

        config.image.enable_augmentation = not getattr(args, "no_augment", False)
        config.image.avatar_use_api = getattr(args, "avatar_api", False)
        config.image.output_quality = getattr(args, "image_quality", 92)

        config.num_workers = getattr(args, "workers", 4)
        config.batch_size = getattr(args, "batch_size", 10)

        config.debug_mode = getattr(args, "debug", False)
        config.dry_run = getattr(args, "dry_run", False)

        return config
    

PROMPT_TEMPLATES = {
    "aus_passport": {
        "has_portrait_photo": True,
        "photo_instructions": (
            "Generate ONE highly realistic portrait of a {target_gender} (age: {target_dob}).\n"
            "Seamlessly integrate this same face into all 3 designated photo slots: MAIN, GHOST (top-right), and HOLOGRAM (bottom-right)."
        ),
        "info": {
            "text_fields": (
                "## 1-TO-1 INPAINTING CONSTRAINTS\n"
                "You are an exact text replacement engine.\n"
                "1. Locate each value from BASE_JSON on the image.\n"
                "2. Erase the old value perfectly, preserving the underlying guilloche patterns.\n"
                "3. Render the corresponding value from TARGET_DATA in the exact same spatial position, using the same font and color.\n"
                "4. CRITICAL DOCUMENT NUMBER REPLACEMENT: You MUST locate and replace the 'document_number' in FOUR specific locations. The target string is exactly 9 characters long. Do NOT duplicate characters or add extra digits.\n"
                "   - LOCATION A (Vertical Perforation): The dot-matrix laser perforated number on the far left edge of the top page.\n"
                "   - LOCATION B (Slanted Text): The slanted/angled micro-printed document number near the bottom edge of the top page's illustration.\n"
                "   - LOCATION C (Top Right Pattern): The small printed number embedded in the background pattern on the top-right page.\n"
                "   - LOCATION D (Main Data Page): The large value under the 'Document No.' label on the middle-right of the bottom page.\n"
                "5. Ignore the MRZ lines at the bottom."
            )
        }
    },
    "aus_medicare_card": {
        "date_format": "Format expiry_date exactly as DD/MM/YYYY or MM/YYYY depending on template.",
        "has_portrait_photo": False,
        "photo_instructions": "CRITICAL: This document DOES NOT contain any portrait or photo slot.",
        "info": {
            "text_fields": (
                "## 1-TO-1 INPAINTING CONSTRAINTS\n"
                "You are an exact text replacement engine.\n"
                "1. PRESERVE BACKGROUND: Maintain the green/yellow background colors and 'medicare' watermark text pattern 100% intact.\n"
                "2. Locate the values in BASE_JSON (medicare_card_number, cardholders block, expiry_date) on the image.\n"
                "3. Erase the old values completely.\n"
                "4. Render the corresponding values from TARGET_DATA in the exact same spatial positions.\n"
                "5. For 'cardholders', render the multiline string exactly, respecting line breaks (\\n) and left-alignment."
            )
        }
    }
}