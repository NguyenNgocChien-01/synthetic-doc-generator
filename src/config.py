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
        "Generate ONE new fictional human face in the MAIN portrait photo slot. "
        "DO NOT copy the face from the template. "
        "Preserve all background patterns, watermarks, and security features exactly."
    ),
    "info": {
        "mrz_instructions": (
            "CRITICAL: You MUST render the MRZ zone at the bottom with EXACTLY the values from TARGET DATA. "
            "DO NOT copy MRZ from the template image. "
            "MRZ LINE 1 (top row): Render EXACTLY: {mrz_line1} — 44 characters, OCR-B monospace font, black text on light background. "
            "MRZ LINE 2 (bottom row): Render EXACTLY: {mrz_line2} — 44 characters, OCR-B monospace font, black text on light background. "
            "Every character must be visible and accurate. < is a filler character, render it as the symbol <."
        ),
        "text_fields": (
            "CRITICAL TEXT SUBSTITUTION ACTION: You must operate as a strict text replacement engine. "
            "Do not alter the layout, positions, fonts, or colors. Locate the exact text string from BASE_JSON "
            "on the template image and overwrite it with the corresponding string from TARGET_DATA.\n\n"
            
            "MAPPING SUBSTITUTION RULES:\n"
            "- Find BASE_JSON 'document_number' value -> Replace with TARGET_DATA 'document_number' (Apply identically across all 4 locations: Laser dots, Top page bottom-right, Bottom page top-right, and MRZ).\n"
            "- Find BASE_JSON 'family_name' value -> Replace with TARGET_DATA 'family_name'.\n"
            "- Find BASE_JSON 'given_names' value -> Replace with TARGET_DATA 'given_names'.\n"
            "- Find BASE_JSON 'nationality' value -> Replace with TARGET_DATA 'nationality'.\n"
            "- Find BASE_JSON 'date_of_birth' value -> Replace with TARGET_DATA 'date_of_birth' (Ensure strict DD MMM YYYY format).\n"
            "- Find BASE_JSON 'sex' value -> Replace with TARGET_DATA 'sex'.\n"
            "- Find BASE_JSON 'place_of_birth' value -> Replace with TARGET_DATA 'place_of_birth'.\n"
            "- Find BASE_JSON 'date_of_issue' value -> Replace with TARGET_DATA 'date_of_issue' (Ensure strict DD MMM YYYY format).\n"
            "- Find BASE_JSON 'date_of_expiry' value -> Replace with TARGET_DATA 'date_of_expiry' (Ensure strict DD MMM YYYY format).\n"
            "- Find BASE_JSON 'authority' value -> Replace with TARGET_DATA 'authority'.\n"
            "- Find BASE_JSON 'signature' text/area -> Erase completely and render a new cursive text reading exactly: TARGET_DATA 'given_names' + space + 'family_name'.\n\n"
            
            "STRICT EXECUTION CONSTRAINTS:\n"
            "1. 1-TO-1 ALIGNMENT: The new text must occupy the exact same spatial boundaries as the old text. Do not shift text coordinates.\n"
            "2. BACKGROUND PRESERVATION: Keep the underlying security textures, guilloche patterns, and watermarks 100% intact. Only modify the text characters.\n"
            "3. NULL HANDLING: If any field in TARGET_DATA is empty or 'null', completely remove/erase the corresponding text value of BASE_JSON from the image without leaving artifacts."
        ),
    }
},
    "driver_license_act": {
        "date_format": "CRITICAL: Format all dates exactly as DD MMM YYYY (e.g., '09 SEP 1986' or '14 JUL 2033'). Do NOT use DD/MM/YYYY.",
        "has_portrait_photo": True,
        "photo_instructions": (
            "Front card layout:\n"
            "- LEFT 60%: text fields only in this order: CITIZEN, full name, address, "
            "Date of Birth, Licence No., Licence Class, Conditions, signature.\n"
            "- RIGHT 40% bottom half: ONE portrait photo only.\n"
            "- TOP RIGHT: ACT Government logo only. NOT a photo slot.\n"
            "- No ghost photos. No duplicate addresses. No invented fields."
        ),
        "info": {
            "name_and_address": "Names and Addresses must be printed in BLACK without any field labels (DO NOT write 'Name:' or 'Address:'). Family name is first and UPPERCASE.",
            "purple_text": "CRITICAL: The values for Date of Birth, Licence No., Expires, Class, and Conditions MUST be printed in PURPLE color.",
            "vertical_card_number": "CRITICAL: The card_number (e.g., F987654321) MUST be rotated exactly 90-degrees vertically (reading bottom-to-top) and placed inside the solid red-bordered box. Do not print it horizontally.",
            "data_strictness": "If a field is 'null', leave the physical area blank."
            "CRITICAL: The authority field and ALL fields above the MRZ zone must be fully visible. "
            "The bottom 9% of the image is reserved for MRZ — do NOT place any data fields there.\n"
        },
    },
    "aus_medicare_card": {
            "date_format": "Format expiry_date exactly as DD/MM/YYYY or MM/YYYY depending on template.",
            "has_portrait_photo": False,
            "photo_instructions": (
                "CRITICAL: This document DOES NOT contain any portrait or photo slot. "
                "DO NOT add any portrait, face, avatar, silhouette, or human figure."
            ),
            "info": {
                "text_fields": (
                    "CRITICAL TEXT SUBSTITUTION ACTION: You must operate as a precise text replacement and automatic multi-line layout engine.\n\n"
                    "MAPPING SUBSTITUTION RULES:\n"
                    "- Find BASE_JSON 'medicare_card_number' -> Replace with TARGET_DATA 'medicare_card_number'.\n"
                    "- Find BASE_JSON 'expiry_date' -> Replace with TARGET_DATA 'expiry_date'.\n"
                    "- CARDHOLDER LIST MULTI-LINE AUTO-ALIGNMENT:\n"
                    "  1. Use the 'cardholders' string in BASE_JSON as the absolute physical anchor on the image to detect font size, typeface, color, and starting X-coordinate.\n"
                    "  2. Completely erase the base text line while preserving 100% of the security background texture.\n"
                    "  3. Parse the TARGET_DATA 'cardholders' string. It contains multiple pre-formatted lines separated by newline characters (\\n), with serial numbers already included at the beginning of each line.\n"
                    "  4. GRID ALIGNMENT CONSTRAINT: Mathematically align all rendered lines strictly to the left, locking them to the exact same X-coordinate as the anchor line. Maintain a rigid, perfectly uniform vertical offset (Y-spacing) between each line to prevent drifting, tilting, or overlapping.\n"
                    "  5. Font weight, kerning, and tracking must remain completely consistent across all lines. Do not alter background graphics."
                "STRICT EXECUTION CONSTRAINTS:\n"
                "1. NO FLOATING TEXT: You are strictly prohibited from generating any extra numbers, characters, or lines between the 'medicare_card_number' and the first line of 'cardholders'. The space between them must remain completely clean.\n"
                "2. PRESERVE BACKGROUND: Maintain the blue background color and security patterns 100% intact behind the new text. Do not blur."
                )
            }
        },
    "default": {
        "date_format": "Format all dates exactly as they appear in the provided JSON data.",
        "has_portrait_photo": False,
        "photo_instructions": "Place the newly generated fictional face in the designated portrait photo slot.",
        "info": {
            "strict_copy": "Copy all text exactly from the JSON. Do not invent missing data. 'null' means leave blank."
        }
    }
}
