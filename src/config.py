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
    "- DATA: 100% rely on JSON. DO NOT invent. If null -> leave blank, erase old text.\n"
    "- NESTED: Extract from nested objects ('front.holder', 'front.licence', etc.).\n"
    "- ARRAY: Concatenate without brackets (['S','B'] -> 'SB').\n"
    "- DATES FRONT=BACK: Dates on front MUST exactly match dates on back.\n"
    "- DOB WATERMARK: If 'dob_watermark' in JSON -> render ONCE clearly. If null -> wipe clean. NEVER duplicate/hallucinate.\n"
    "- CONDITIONS: Render exactly from JSON. If null -> leave blank.\n"
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

    "driver_license": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": _DL_SHARED_PHOTO,
        "info": {"shared": _DL_SHARED_INFO}
    },
    
"driver_license_act": {
    "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
    "photo_instructions": (
        "Front card layout:\n"
        "- LEFT 60%: text fields only in this order: CITIZEN, full name, address, "
        "Date of Birth, Licence No., Licence Class, Conditions, signature.\n"
        "- RIGHT 40% bottom half: ONE portrait photo only.\n"
        "- TOP RIGHT: ACT Government logo only. NOT a photo slot.\n"
        "- No ghost photos. No duplicate addresses. No invented fields."
    ),
    "info": {"shared": _DL_SHARED_INFO}
},
    # ── NSW 
    "driver_license_nsw": {
        "date_format": "CRITICAL: Format all dates exactly as DD/MM/YYYY.",
        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "NSW SPECIFIC:\n"
            "- Background: bold RED top band, white body, gold/red diagonal security stripe pattern.\n"
            "- NSW Waratah logo = NOT a photo slot.\n"
            "- Holographic overlay on card surface -> preserve as-is.\n"
        ),
        "info": {"shared": _DL_SHARED_INFO}
    },
 
    # ── VIC 
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
# ── QLD ───
    "driver_license_qld": {
        "date_format": (
            "CRITICAL DATE FORMATS — no exceptions:\n"
            "- DOB on front: 'DD Mmm YYYY' e.g. '03 Feb 1994'\n"
            "- Class table Effective + Expiry: 'DD.MM.YY' e.g. '15.12.22'\n"
            "- Back DOB field: 'DOB.MM.YYYY' e.g. 'DOB.02.1994' — dot after DOB, dot between MM and YYYY\n"
            "- Back age indicator: 'MM-YY' e.g. '02-94' — NO word 'Age', just digits\n"
            "- Back Spiicstate: state abbrev + space + 'DD.MM.YYYY' e.g. 'QLD 11.02.1994'\n"
            "- Card number: NO spaces, starts with D e.g. 'D463514007'\n"
            "- Licence number: NO spaces e.g. '65604018'\n"
        ),

        "layout_front": (
            "QLD DRIVER LICENCE FRONT — gold/yellow guilloche background.\n\n"

            "TOP BAR:\n"
            "  LEFT: 'Driver Licence' dark red italic\n"
            "  CENTER: 'Australia' large faint diagonal watermark\n"
            "  RIGHT: 'LICENCE NO.' label then number — NO spaces in number\n\n"

            "LEFT COLUMN (top to bottom):\n"
            "  1. Surname on line 1 — exactly once, bold large\n"
            "  2. Given name on line 2 — exactly once, bold large\n"
            "  3. EMV smart card chip (gold) — never cover\n"
            "  4. Ghost portrait — grayscale, faded opacity ~25%, small\n"
            "  5. Handwritten signature over ghost photo\n"
            "  6. 'Queensland, Australia' tiny italic at bottom\n\n"

            "CENTER COLUMN (top to bottom):\n"
            "  1. Full name bold — print ONCE only, no repetition\n"
            "  2. 'Sex' label on same line as name, right aligned\n"
            "  3. DOB: 'DD Mmm YYYY' — MANDATORY, never omit\n"
            "  4. Class table — 4 columns exactly:\n"
            "       Class | Type | Effective | Expiry\n"
            "       C       O      DD.MM.YY   DD.MM.YY\n"
            "       RE      L      DD.MM.YY   DD.MM.YY\n"
            "     Both Effective AND Expiry columns MUST be filled\n"
            "  5. Address line 1 (unit/flat/street number street name)\n"
            "  6. Suburb STATE Postcode\n"
            "  7. 'Conditions X' where X is condition code\n"
            "  8. 'Drive safely' italic small\n\n"

            "RIGHT COLUMN:\n"
            "  Full portrait photo — color, full face visible, NOT cropped\n"
            "  Signature over bottom edge of photo\n\n"

            "BOTTOM RIGHT: Queensland Government logo + seal\n\n"

            " NEVER:\n"
            "  - Repeat name more than once in center column\n"
            "  - Leave DOB blank\n"
            "  - Leave Effective date blank\n"
            "  - Add spaces to licence number\n"
            "  - Add 'Height' field\n"
            "  - Crop the main photo\n"
        ),

        "layout_back": (
            "QLD DRIVER LICENCE BACK — silver/grey, faint Australia map background.\n\n"

            "ZONE 1 — TOP LEFT (barcode area):\n"
            "  - Wide horizontal barcode (Code 128), flush top\n"
            "  - Barcode number text below it: 'ABnoteNZ' + 10 digits\n\n"

            "ZONE 2 — TOP RIGHT (info block), same height as barcode:\n"
            "  Two columns side by side:\n"
            "  Left sub-col:      Right sub-col:\n"
            "  'Expiry Date'      'Spiicstate' (= 'QLD DD.MM.YYYY')\n"
            "  'DOB.MM.YYYY'      (empty)\n"
            "  'MM-YY'            (empty)\n\n"

            "ZONE 3 — MIDDLE LEFT:\n"
            "  'View Terms of use and update your information at:'\n"
            "  'www.tmr.qld.gov.au'\n"
            "   NO address, NO name here\n\n"

            "ZONE 4 — MIDDLE RIGHT:\n"
            "  Holographic Australia map sticker (colorful iridescent)\n\n"

            "ZONE 5 — BOTTOM:\n"
            "  LEFT: small vertical barcode (rotated 90°) + barcode digits\n"
            "  RIGHT: 'Card number' label, value below — NO spaces, starts with D\n\n"

            " NEVER:\n"
            "  - Put address or name anywhere on back\n"
            "  - Put barcode at bottom instead of top\n"
            "  - Use 'Age XX-XX' — just 'MM-YY' digits only\n"
            "  - Use 'DOB DD-MM-YYYY' — must be 'DOB.MM.YYYY'\n"
            "  - Add spaces to card number\n"
            "  - Show two 'Card number' fields\n"
        ),

        "photo_instructions": (
            _DL_SHARED_PHOTO +
            "QLD SPECIFIC:\n"
            "- MAIN PHOTO: color, full face NOT cropped, right column\n"
            "- GHOST PHOTO: same face, grayscale + ~25% opacity, bottom left, smaller\n"
            "- SIGNATURE: handwritten cursive, over ghost photo AND over main photo bottom\n"
            "- EMV chip: gold, left center, never covered\n"
            "- QLD Government logo: bottom right always\n"
        ),

        "info": {"shared": _DL_SHARED_INFO}
    },
    # ── WA ──
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
 
    # ── SA ────
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
            "members": "Render 'members' array sequentially from position 1, full_name only. Fewer members than template -> leave remaining lines blank (do NOT copy from template).",
            "data": "DO NOT invent/infer data. If null -> leave blank.",
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
