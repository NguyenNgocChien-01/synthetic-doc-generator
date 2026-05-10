"""
Module cấu hình hệ thống toàn cục.
Tập trung quản lý tất cả tham số cấu hình của công cụ sinh dữ liệu.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# Các nền tảng API được hỗ trợ
PLATFORM_GOOGLE_AI_STUDIO = "google_ai_studio"
PLATFORM_VERTEX_AI = "vertex_ai"
PLATFORM_OPENAI = "openai"
PLATFORM_NONE = "none"

SUPPORTED_PLATFORMS = [
    PLATFORM_GOOGLE_AI_STUDIO,
    PLATFORM_VERTEX_AI,
    PLATFORM_OPENAI,
    PLATFORM_NONE,
]

# Các loại tài liệu được hỗ trợ
SUPPORTED_DOC_TYPES = [
    "passport",
    "driver_license",
    "medicare_card",
    "utility_bill",
]

# Cấu hình mặc định theo từng nền tảng
PLATFORM_ENDPOINTS: Dict[str, str] = {
    PLATFORM_GOOGLE_AI_STUDIO: "https://generativelanguage.googleapis.com/v1beta",
    PLATFORM_VERTEX_AI: "https://{region}-aiplatform.googleapis.com/v1",
    PLATFORM_OPENAI: "https://api.openai.com/v1",
}

# Giới hạn mặc định theo nền tảng (yêu cầu/phút)
PLATFORM_RATE_LIMITS: Dict[str, int] = {
    PLATFORM_GOOGLE_AI_STUDIO: 60,
    PLATFORM_VERTEX_AI: 300,
    PLATFORM_OPENAI: 60,
    PLATFORM_NONE: 0,
}

# Giới hạn token mặc định mỗi ngày theo nền tảng
PLATFORM_DAILY_TOKEN_LIMITS: Dict[str, int] = {
    PLATFORM_GOOGLE_AI_STUDIO: 1_000_000,
    PLATFORM_VERTEX_AI: 10_000_000,
    PLATFORM_OPENAI: 500_000,
    PLATFORM_NONE: 0,
}


@dataclass
class APIConfig:
    """Cấu hình kết nối API."""
    platform: str = PLATFORM_NONE
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    project_id: Optional[str] = None
    region: str = "us-central1"
    model_name: str = "gemini-1.5-flash"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    retry_backoff_factor: float = 2.0

    def resolve_endpoint(self) -> str:
        """Phân giải endpoint dựa trên nền tảng hoặc dùng endpoint tùy chỉnh."""
        if self.endpoint:
            return self.endpoint
        base = PLATFORM_ENDPOINTS.get(self.platform, "")
        if self.platform == PLATFORM_VERTEX_AI:
            return base.format(region=self.region)
        return base


@dataclass
class QuotaConfig:
    """Cấu hình quản lý hạn mức tài nguyên."""
    daily_token_limit: int = 1_000_000
    requests_per_minute: int = 60
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0
    quota_check_interval_seconds: int = 60


@dataclass
class ImageConfig:
    """Cấu hình xử lý và sinh ảnh."""
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
    """Cấu hình lưu trữ dữ liệu."""
    templates_dir: str = "templates"
    dataset_dir: str = "dataset"
    log_dir: str = "logs"
    use_uuid_filenames: bool = True
    id_prefix: str = ""
    id_zero_padding: int = 6


@dataclass
class Config:
    """Cấu hình tổng hợp của toàn bộ hệ thống."""
    api: APIConfig = field(default_factory=APIConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Cấu hình đa luồng
    num_workers: int = 4
    batch_size: int = 10

    # Chế độ gỡ lỗi
    debug_mode: bool = False
    dry_run: bool = False

    @classmethod
    def from_cli_args(cls, args: Any) -> "Config":
        """
        Khởi tạo cấu hình từ các đối số dòng lệnh.

        Tham số:
            args: Đối tượng chứa các đối số đã phân tích từ CLI.

        Trả về:
            Đối tượng Config đã được cấu hình.
        """
        config = cls()

        # Cấu hình API
        platform = getattr(args, "api", PLATFORM_NONE)
        config.api.platform = platform
        config.api.api_key = (
            getattr(args, "api_key", None)
            or os.environ.get("API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        config.api.endpoint = getattr(args, "endpoint", None)
        config.api.project_id = (
            getattr(args, "project_id", None)
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        config.api.region = getattr(args, "region", "us-central1")
        config.api.model_name = getattr(args, "model", "gemini-1.5-flash")
        config.api.max_retries = getattr(args, "max_retries", 3)

        # Cấu hình hạn mức theo nền tảng
        config.quota.daily_token_limit = PLATFORM_DAILY_TOKEN_LIMITS.get(
            platform, 1_000_000
        )
        config.quota.requests_per_minute = PLATFORM_RATE_LIMITS.get(platform, 60)

        # Cấu hình lưu trữ
        config.storage.templates_dir = getattr(args, "templates_dir", "templates")
        config.storage.dataset_dir = getattr(args, "output_dir", "dataset")

        # Cấu hình xử lý ảnh
        config.image.enable_augmentation = not getattr(args, "no_augment", False)
        config.image.avatar_use_api = getattr(args, "avatar_api", False)
        config.image.output_quality = getattr(args, "image_quality", 92)

        # Cấu hình đa luồng
        config.num_workers = getattr(args, "workers", 4)
        config.batch_size = getattr(args, "batch_size", 10)

        # Chế độ đặc biệt
        config.debug_mode = getattr(args, "debug", False)
        config.dry_run = getattr(args, "dry_run", False)

        return config
