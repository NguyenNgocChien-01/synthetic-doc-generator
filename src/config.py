# src/config.py
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

PLATFORM_VERTEX_AI = "vertex_ai"
SUPPORTED_PLATFORMS = [PLATFORM_VERTEX_AI]

SUPPORTED_DOC_TYPES = [
    "passport",
    "driver_license",
    "medicare_card",
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
    "passport": {
        "date_format": "ALL dates MUST exactly match the format provided in the JSON (e.g., '1983-07-29'). Do not reformat to 'DD-MM-YYYY' or 'DD MMM YYYY'.",
        "photo_instructions": "Place the newly generated fictional face ONLY in the MAIN/LARGEST portrait photo slot. For any secondary 'ghost' or holographic portrait slots, leave the template's default appearance intact.",
        "info": {
            "mrz_handling": "(44 characters) Maintain the Machine Readable Zone (MRZ) structure at the bottom. Replace the text using the provided data while keeping the chevron '<' alignment and spacing exactly like the template.",
            "data_strictness": "If a field in the JSON is 'null', 'None', or empty, leave that corresponding physical area blank on the document."
        }
    },
    "driver_license": {
        "date_format": "CRITICAL: The JSON dates are in ISO format (YYYY-MM-DD). You MUST format them exactly as YYYY-MM-DD on the generated document image. DO NOT change them to DD/MM/YYYY.",
        "photo_instructions": (
            "Generate ONE new fictional human face, ensuring it does not resemble anyone in the original template. "
            "The face MUST be created from scratch, matching the provided JSON details including age, gender, and stature. "
            "CRITICAL VISUAL INSPECTION - Examine the provided template carefully to determine portrait slots. "
            "CRITICAL ACT WARNING: The white text and logo area containing the words 'ACT Government' and the coat of arms is NOT a photo slot. This is sacred text/logo data and MUST NOT be modified, replaced, or covered by a face image. Only modify the single, clearly defined portrait photo frame. Delete any other perceived 'extra' photo elements, but do not touch the 'ACT Government' text/logo data. "
            "CRITICAL PHOTO OVERLAYS: You MUST preserve all state-specific security features, holograms, transparent watermark numbers, guilloche lines, AND ANY handwritten signature or text overlapping the portrait areas. If a signature overlaps a photo, the signature MUST appear *on top* of the newly generated face."
        ),
        
        "info": {
            "nested_json_handling": "The JSON structure uses nested objects like 'front.holder', 'front.licence', and 'front.address'. Extract the values from these objects and apply them to the correct sections on the document.",
            "array_formatting": "If fields like 'licence_classes' or 'conditions' are arrays, concatenate the elements into a single string without brackets. Example: ['S', 'B'] renders as 'SB' or 'S B' depending on the template spacing.",
            "dynamic_data_rendering": "DO NOT invent, infer, or calculate any data. Rely 100% on the JSON. If a key is 'null', leave its designated space entirely blank.",
            "front_back_consistency": "Render 'conditions_legend' or 'transport_notice' exactly as provided in the 'back' object. Ensure dates on the Front strictly match dates on the Back.",
            "variable_watermarks": "Render 'dob_watermark', 'age_indicator', or 'card_number' matching the size, opacity, and specific location shown in the template. If 'null', remove them.",
            "stature_handling": "You MUST generate a unique character that is physically distinct. A person specified as 'short boy' or with a child/adolescent age must be a smaller character with appropriate stature and features for that character type, not just a small-adult version. Use height data and name-to-gender logic to create a distinct individual."
        }
    },
    "default": {
        "date_format": "Format all dates exactly as they appear in the provided JSON data.",
        "photo_instructions": "Place the newly generated fictional face in the designated portrait photo slot.",
        "info": {
            "strict_copy": "Copy all text exactly from the JSON. Do not invent missing data. 'null' means leave blank."
        }
    }
}