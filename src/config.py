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
    


_DL_SHARED_INFO = (
    "SHARED RULES (apply to ALL states):\n"
    "- DATA: 100% rely on JSON. DO NOT invent. If null → leave blank, erase old text.\n"
    "- NESTED: Extract from nested objects ('front.holder', 'front.licence', etc.).\n"
    "- ARRAY: Concatenate without brackets (['S','B'] → 'SB').\n"
    "- DATES FRONT=BACK: Dates on front MUST exactly match dates on back.\n"
    "- DOB WATERMARK: If 'dob_watermark' in JSON → render ONCE clearly. If null → wipe clean. NEVER duplicate/hallucinate.\n"
    "- CONDITIONS: Render exactly from JSON. If null → leave blank.\n"
)
 
_DL_SHARED_PHOTO = (
    "PHOTO RULES:\n"
    "- Generate ONE fictional human face. Do NOT copy or resemble anyone in the template.\n"
    "- Place ONLY in the clearly designated portrait slot (usually a rectangular box with a visible border).\n"
    "- LOGOS, COATS OF ARMS, STATE EMBLEMS are NOT photo slots — do NOT place any face over them.\n"
    "- GHOST/HOLOGRAM layers that are part of the security design must be preserved as-is unless stated otherwise.\n"
    "- CLEANUP: Erase ALL ghosted numbers, duplicated dates, or floating artefacts around the photo. Background must be clean.\n"
)
 

PROMPT_TEMPLATES = {
    "passport": {
        "date_format": "ALL dates MUST exactly match the format provided in the JSON (e.g., '1983-07-29'). Do not reformat to 'DD-MM-YYYY' or 'DD MMM YYYY'.",
        "photo_instructions": "Place the newly generated fictional face ONLY in the MAIN/LARGEST portrait photo slot. For any secondary 'ghost' or holographic portrait slots, leave the template's default appearance intact.",
        "info": {
            "mrz_handling": "(44 characters) Maintain the Machine Readable Zone (MRZ) structure at the bottom. Replace the text using the provided data while keeping the chevron '<' alignment and spacing exactly like the template.",
            "data_strictness": "If a field in the JSON is 'null', 'None', or empty, leave that corresponding physical area blank on the document."
        }
    },
     # ── DRIVER LICENSE fallback (no state) ───────────────────────────────────
    "driver_license": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": _DL_SHARED_PHOTO,
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── ACT ──────────────────────────────────────────────────────────────────
    "driver_license_act": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "ACT SPECIFIC:\n"
            "- Background: WHITE with red/dark-blue header bar.\n"
            "- 'ACT Government' text + Coat of Arms in header = NOT a photo slot. Do NOT touch it.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── NSW ──────────────────────────────────────────────────────────────────
    "driver_license_nsw": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "NSW SPECIFIC:\n"
            "- Background: bold RED top band, white body, gold/red diagonal security stripe pattern.\n"
            "- NSW Waratah logo = NOT a photo slot.\n"
            "- Holographic overlay on card surface → preserve as-is.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── VIC ──────────────────────────────────────────────────────────────────
    "driver_license_vic": {
        "date_format": "CRITICAL: Format all dates exactly as DD-MM-YYYY (e.g., 30-03-2030) to match the template. DO NOT use YYYY-MM-DD or DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "VIC SPECIFIC RULES:\n"
            "- PHOTO SLOT: 1 photo on FRONT ONLY (right side). The back MUST NOT contain any face or portrait artifacts.\n"
            "- STRICT BOUNDARIES: The portrait MUST stay strictly within its designated rectangular box. DO NOT let the face, hair, or clothing bleed into the green background or overlap text.\n"
            "- CRITICAL CLEANUP: You MUST completely ERASE any hallucinated ghost text, messy overlapping dates, or scrambled letters (e.g., 'DAPIRY', 'EOXNRY') on both front and back. The background must remain a clean green chevron pattern.\n"
            "- FRONT DOB WATERMARK: If JSON has 'dob_watermark', render it exactly ONCE as crisp, clean text overlapping the bottom edge of the photo. Do NOT duplicate or smudge it. If null, wipe the area clean.\n"
            "- BACK WATERMARKS: Preserve the large background numbers (expiry/DOB) on the back, but render them cleanly without scrambling the surrounding text."
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
    # ── QLD ──────────────────────────────────────────────────────────────────
    "driver_license_qld": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "QLD SPECIFIC:\n"
            "- Background: WHITE/LIGHT with MAROON top band.\n"
            "- QLD has a semi-transparent ghost portrait + holographic overlay printed ON TOP of the main photo.\n"
            "  → New face goes in MAIN portrait slot only. Ghost/hologram layer sits on top → preserve exactly as template.\n"
            "- Queensland Government logo + Coat of Arms = NOT photo slots.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── WA ───────────────────────────────────────────────────────────────────
    "driver_license_wa": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "WA SPECIFIC:\n"
            "- Background: YELLOW/GOLD top section with BLACK text, lighter body.\n"
            "- 'Government of Western Australia' text + Black Swan emblem = NOT photo slots.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── SA ───────────────────────────────────────────────────────────────────
    "driver_license_sa": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "SA SPECIFIC:\n"
            "- Background: RED top band, white body.\n"
            "- 'Government of South Australia' logo = NOT a photo slot.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── TAS ──────────────────────────────────────────────────────────────────
    "driver_license_tas": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "TAS SPECIFIC:\n"
            "- Background: GREEN top band, white/light body.\n"
            "- 'Service Tasmania' logo + Tasmanian Devil emblem = NOT photo slots.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── NT ───────────────────────────────────────────────────────────────────
    "driver_license_nt": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            "PHOTO RULES:\n"
            "- NT license has TWO portrait slots: 1 MAIN (larger) + 1 SECONDARY (smaller).\n"
            "- Generate ONE fictional person and place the SAME face in BOTH slots (same person, different crop/size).\n"
            "- Do NOT generate two different faces.\n"
            "- 'Northern Territory Government' logo = NOT a photo slot.\n"
            "- Background: ORANGE/OCHRE top band, white body.\n"
            "- CLEANUP: Erase ghosted numbers/watermarks around BOTH photo areas.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── MEDICARE ─────────────────────────────────────────────────────────────
    "medicare_card": {
        "date_format": "Format 'expiry_date' as MM/YYYY (e.g. 03/2028). Place next to 'VALID TO' text.",
        "photo_instructions": (
            "CRITICAL: Medicare cards have NO portrait photo.\n"
            "DO NOT generate, add, or draw any human face, avatar, or ghost image anywhere on this card."
        ),
        "info": {
            "background": "Preserve the repeating 'medicare' watermark background pattern — do NOT erase or smudge it.",
            "card_number": "11-digit number. Render with spacing: 4 digits, space, 5 digits, space, 1 digit.",
            "members": "Render 'members' array sequentially from position 1, full_name only. Fewer members than template → leave remaining lines blank (do NOT copy from template).",
            "data": "DO NOT invent/infer data. If null → leave blank.",
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
