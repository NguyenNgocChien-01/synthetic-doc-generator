
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

PLATFORM_VERTEX_AI = "vertex_ai"
SUPPORTED_PLATFORMS = [PLATFORM_VERTEX_AI]

SUPPORTED_DOC_TYPES = [
    "aus_passport",
    "aus_driver_license",
    "aus_medicare_card",
    "utility_bill",
    "aus_energy_bill",
    "aus_wwc_card"
]

PLATFORM_ENDPOINTS: Dict[str, str] = {
    PLATFORM_VERTEX_AI: "https://{region}-aiplatform.googleapis.com/v1",
}

#  Gemini model use :generateContent
VERTEX_STANDARD_MODELS = {
    "gemini-2.5-flash-image"
}
# preview (dont use)
AGENT_PLATFORM_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
}

# text --> Img, no using
# Imagen model use :predict
# IMAGEN_MODELS = {
#     "imagen-3.0-generate-001",
#     "imagen-3.0-fast-generate-001",
# }

ALL_IMAGE_MODELS = VERTEX_STANDARD_MODELS | AGENT_PLATFORM_MODELS 
# |  IMAGEN_MODELS


def get_model_endpoint(region: str, project_id: str, model: str) -> str:
    if model in AGENT_PLATFORM_MODELS:
        base = f"https://{region}-aiplatform.googleapis.com/v1beta1"
    else:
        base = f"https://{region}-aiplatform.googleapis.com/v1"
        
    
    # action = "predict" if "imagen" in model.lower() else "generateContent"
    action = "generateContent"
    
    return (
        f"{base}/projects/{project_id}/locations/{region}"
        f"/publishers/google/models/{model}:{action}"
    )


@dataclass
class APIConfig:
    platform: str = PLATFORM_VERTEX_AI
    endpoint: Optional[str] = None
    project_id: Optional[str] = None
    region: str = "us-central1"
    model_name: str = "gemini-2.5-flash-image"
    image_model: str = "gemini-2.5-flash-image"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    retry_backoff_factor: float = 2.0

    def resolve_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint
        base = PLATFORM_ENDPOINTS.get(self.platform, "")
        return base.format(region=self.region)

    def get_image_endpoint(self) -> str:
        """return endpoint --> image model"""
        if self.endpoint:
            # action = "predict" if "imagen" in self.image_model.lower() else "generateContent"
            action = "generateContent"
            return f"{self.endpoint}/{self.image_model}:{action}"
        return get_model_endpoint(self.region, self.project_id or "", self.image_model)

    def is_agent_platform_model(self) -> bool:
        return self.image_model in AGENT_PLATFORM_MODELS


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

        config.api.platform   = PLATFORM_VERTEX_AI
        config.api.endpoint   = getattr(args, "endpoint", None)
        config.api.project_id = (
            getattr(args, "project_id", None)
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        config.api.region      = getattr(args, "region", "us-central1")
        config.api.image_model = getattr(args, "image_model", "gemini-2.5-flash-image")
        config.api.model_name  = config.api.image_model
        config.api.max_retries = getattr(args, "max_retries", 3)

        config.storage.templates_dir = getattr(args, "templates_dir", "templates")
        config.storage.dataset_dir   = getattr(args, "output_dir", "dataset")

        config.image.enable_augmentation = not getattr(args, "no_augment", False)
        config.image.avatar_use_api      = getattr(args, "avatar_api", False)
        config.image.output_quality      = getattr(args, "image_quality", 92)

        config.num_workers = getattr(args, "workers", 4)
        config.batch_size  = getattr(args, "batch_size", 10)
        config.debug_mode  = getattr(args, "debug", False)
        config.dry_run     = getattr(args, "dry_run", False)

        return config
    
    

PROMPT_TEMPLATES = {
    "aus_passport": {
        "photo_mode": "MULTIPLE", 
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
        "photo_mode": "NONE",
        "photo_instructions": "CRITICAL: This document DOES NOT contain any portrait or photo slot.",
        "info": { 
            "text_fields": (
                "Replace ONLY text fields (medicare_card_number, cardholders, expiry_date). "
                "Preserve background, watermark, font style, color, and layout exactly. "
                "Cardholders: multiline, evenly spaced rows, columns aligned."
            )
        }
    },
    "driver_license/act": {
        "date_format": "Format dates exactly as DD MMM YYYY (e.g., 13 FEB 1990) for ALL .",
        "photo_mode": "SINGLE",  
        "photo_instructions": (
            "Preserve the existing portrait placement and frame on the middle right of the front card. "
            "Generate a highly realistic driver license style front-facing photo of a {target_gender} (age: {target_dob}). "
            "Generate a random black ink cursive signature on the bottom left of the front card."
        ),
        "info": {
            "text_fields": (
                "You are an exact image editing and text replacement engine.\n"
                "STRICT RULES:\n"
                "1. PRESERVE EVERYTHING: Maintain original layout, background texture, gradients, kerning, colors, watermarks, holograms, shadows, and compression artifacts 100% intact.\n"
                "2. Do NOT redesign, regenerate, or hallucinate any part of the card.\n"
                "3. Do NOT move or alter any non-text structural element.\n"
                "4. Locate the old text values from BASE_JSON and erase them completely without leaving blur patches or smudge artifacts.\n"
                "5. Render the corresponding new values from TARGET_DATA in the exact same spatial positions.\n"
                "6. TYPOGRAPHY: You MUST match the original font weight, scale, color, and baseline alignment exactly.\n"
                "7. CRITICAL: Do NOT add any QR codes, barcodes, or extra visual elements not present in the original template.\n"
                "8. Do NOT blindly dump all JSON values onto the image. Only replace existing fields according to the layout.\n\n"
                "LAYOUT MAPPING HINTS (FRONT CARD):\n"
                "- Full Name: Top left area under the header.\n"
                "- Address Lines: Below the Full Name.\n"
                "- Date of Birth: Middle left, aligned with 'Date of Birth' label.\n"
                "- Licence No: Below Date of Birth, aligned with 'Licence No.' label.\n"
                "- Licence Class: Below Licence No.\n"
                "- Conditions: Below Licence Class.\n"
                "- DOB Watermark (8 digits): Top right corner.\n"
                "- Vertical Card Number: Printed vertically (rotated -90 degrees) near the right edge of the portrait.\n"
                "- Horizontal Card Number: Bottom left corner, below the signature.\n\n"
            )
        }
    },
    "aus_driver_license/vic": {
        "date_format": "Format dates exactly as DD-MM-YYYY for ALL (eg. 13-02-1990). ",
        "photo_mode": "SINGLE",  
        "photo_instructions": (
            "Preserve the existing portrait placement and frame on the middle right of the front card. "
            "Generate a highly realistic driver license style front-facing photo of a {target_gender} (age: {target_dob}). "
            "Generate a random black ink cursive signature on the bottom left of the front card."
        ),
        "info": {
            "text_fields": (
                "You are an exact image editing and text replacement engine.\n"
                "STRICT RULES:\n"
                "1. PRESERVE EVERYTHING: Maintain original layout, background texture, gradients, kerning, colors, watermarks, holograms, shadows, and compression artifacts 100% intact.\n"
                "2. Do NOT redesign, regenerate, or hallucinate any part of the card.\n"
                "3. Do NOT move or alter any non-text structural element.\n"
                "4. Locate the old text values from BASE_JSON and erase them completely without leaving blur patches or smudge artifacts.\n"
                "5. Render the corresponding new values from TARGET_DATA in the exact same spatial positions.\n"
                "6. TYPOGRAPHY: You MUST match the original font weight, scale, color, and baseline alignment exactly.\n"
                "7. CRITICAL: Do NOT add any QR codes, barcodes, or extra visual elements not present in the original template.\n"
                "8. Do NOT blindly dump all JSON values onto the image. Only replace existing fields according to the layout.\n"
                "9. CRITICAL FIX: Do NOT print JSON keys or descriptive labels (e.g., 'Full Name:', 'Address:'). Render ONLY the target values. Ensure the name is printed exactly once to avoid overlapping with address lines.\n\n"
                "LAYOUT MAPPING HINTS (FRONT CARD):\n"
                "- Name: Top left area under the blue header. (Value only, NO label).\n"
                "- Address Lines: Directly below the Name. (Value only, NO label).\n"
                "- Licence No: Top right area, immediately below the 'LICENCE NO.' label.\n"
                "- Licence Expiry: Middle left, below the 'LICENCE EXPIRY' label.\n"
                "- Date of Birth: Middle area, below the 'DATE OF BIRTH' label.\n"
                "- Licence Type: Bottom left, below the 'LICENCE TYPE' label.\n"
                "- Conditions: Bottom area, below the 'CONDITIONS' label.\n"
                "- Vertical Card Number: Printed vertically (rotated -90 degrees) near the left edge of the portrait.\n\n"
            )
        }
    },
        "aus_wwc_card/vic": {
        "date_format": "Format dates exactly as DD-MM-YYYY for ALL.Not is DD-MMM-YYYY",
        "photo_mode": "SINGLE",  
        "photo_instructions": (
            "Preserve the existing portrait placement and frame on the middle right of the front card. "
            "Generate a highly realistic driver license style front-facing photo of a {target_gender} (age: {target_dob}). "
            "Generate a random black ink cursive signature on the bottom left of the front card."
        ),
        "info": {
            "text_fields": (
                "You are an exact image editing and text replacement engine.\n"

            )
        }
    }
}

