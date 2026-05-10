"""
Module kết nối linh hoạt với nhiều nền tảng AI API.
Hỗ trợ Google AI Studio, Vertex AI và OpenAI.
Tất cả lệnh gọi đều được bảo vệ bởi cơ chế thử lại và kiểm soát tốc độ.
"""

import io
import json
import logging
import random
import time
import base64
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from .config import (
    APIConfig,
    PLATFORM_GOOGLE_AI_STUDIO,
    PLATFORM_VERTEX_AI,
    PLATFORM_OPENAI,
    PLATFORM_NONE,
)
from .rate_limiter import RateLimiter, RateLimitExceededError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Lỗi chung khi gọi API."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class APIUnavailableError(APIError):
    """Lỗi khi API không khả dụng hoặc chưa cấu hình."""
    pass


class BaseAPIProvider(ABC):
    """Lớp cơ sở trừu tượng cho các nhà cung cấp API."""

    @abstractmethod
    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        """Sinh ảnh đại diện từ mô tả văn bản."""
        pass

    @abstractmethod
    def check_quota(self) -> dict:
        """Kiểm tra hạn mức API còn lại."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Kiểm tra API có khả dụng không."""
        pass


class NoAPIProvider(BaseAPIProvider):
    """
    Nhà cung cấp giả - không gọi API thực.
    Dùng khi người dùng chọn --api none hoặc không cấu hình API key.
    """

    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        logger.debug("Che do khong API: bo qua viec sinh anh dai dien qua API.")
        return None

    def check_quota(self) -> dict:
        return {"trang_thai": "khong_su_dung_api", "thong_bao": "Che do khong API"}

    def is_available(self) -> bool:
        return False


class GoogleAIStudioProvider(BaseAPIProvider):
    """Kết nối với Google AI Studio (Gemini API)."""

    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter

        if not config.api_key:
            raise APIUnavailableError(
                "API key chua duoc cau hinh cho Google AI Studio. "
                "Vui long thiet lap bien moi truong GOOGLE_API_KEY hoac dung --api-key."
            )

        self.base_url = config.resolve_endpoint()
        logger.info("Khoi tao Google AI Studio Provider, model: %s.", config.model_name)

    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        """
        Sinh ảnh đại diện sử dụng Gemini Imagen API.
        Hiện tại sử dụng placeholder vì Imagen cần endpoint riêng.
        """
        logger.debug("Goi Google AI Studio de sinh anh dai dien.")
        # Gemini hiện không hỗ trợ sinh ảnh trực tiếp qua API text thông thường
        # Trả về None để dùng avatar placeholder
        return None

    def check_quota(self) -> dict:
        """Kiểm tra hạn mức bằng cách gọi API danh sách model."""
        try:
            url = f"{self.base_url}/models?key={self.config.api_key}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return {
                        "trang_thai": "kha_dung",
                        "thong_bao": "Ket noi thanh cong toi Google AI Studio.",
                    }
        except urllib.error.HTTPError as loi:
            return {
                "trang_thai": "loi",
                "ma_loi": loi.code,
                "thong_bao": f"Loi HTTP: {loi.code}",
            }
        except Exception as loi:
            return {
                "trang_thai": "loi",
                "thong_bao": str(loi),
            }
        return {"trang_thai": "khong_ro"}

    def is_available(self) -> bool:
        result = self.check_quota()
        return result.get("trang_thai") == "kha_dung"

    def _make_request(self, payload: dict) -> dict:
        """Gửi yêu cầu đến Gemini API."""
        url = (
            f"{self.base_url}/models/{self.config.model_name}"
            f":generateContent?key={self.config.api_key}"
        )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as loi:
            error_body = loi.read().decode("utf-8", errors="replace")
            raise APIError(
                f"Loi HTTP {loi.code}: {error_body}", status_code=loi.code
            ) from loi
        except urllib.error.URLError as loi:
            raise APIError(f"Loi ket noi: {loi.reason}") from loi
            
    def generate_document_image(self, template_image: Image.Image, json_data: dict) -> Optional[Image.Image]:
        """
        Gửi ảnh gốc và JSON dữ liệu lên API để yêu cầu sinh ảnh tài liệu mới.
        """
        logger.debug("Goi API sinh anh tai lieu dua tren anh goc va JSON.")
        
        # Chuyển đổi ảnh gốc sang Base64
        buffered = io.BytesIO()
        template_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Chuẩn bị Prompt: Yêu cầu API dựa vào ảnh gốc và thay thế nội dung bằng JSON
        prompt_text = (
            "Dưới đây là một ảnh mẫu của giấy phép lái xe và một tệp dữ liệu JSON. "
            "Hãy tạo ra một bức ảnh giấy phép lái xe chân thực hoàn toàn mới, "
            "giữ nguyên bố cục của ảnh mẫu nhưng thay thế toàn bộ thông tin văn bản "
            "và mã vạch bằng các giá trị được cung cấp trong tệp JSON sau:\n"
            f"{json.dumps(json_data, ensure_ascii=False, indent=2)}"
        )

        # Cấu trúc payload gửi API (Dành cho mô hình hỗ trợ Vision/Image Generation)
        # Lưu ý: Endpoint thực tế phụ thuộc vào mô hình (ví dụ: Imagen 3 hoặc Gemini 1.5 Pro)
        url = f"{self.base_url}/models/{self.config.model_name}:predict?key={self.config.api_key}"
        
        payload = {
            "instances": [
                {
                    "prompt": prompt_text,
                    "image": {
                        "bytesBase64Encoded": img_str
                    }
                }
            ],
            "parameters": {
                "sampleCount": 1
            }
        }
        
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                # Trích xuất ảnh kết quả từ phản hồi
                b64_result = result["predictions"][0]["bytesBase64Encoded"]
                img_data = base64.b64decode(b64_result)
                return Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as loi:
            logger.error("Loi khi goi API sinh anh: %s", loi)
            return None


class VertexAIProvider(BaseAPIProvider):
    """Kết nối với Google Vertex AI."""

    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter

        if not config.project_id:
            raise APIUnavailableError(
                "Project ID chua duoc cau hinh cho Vertex AI. "
                "Vui long thiet lap bien moi truong GOOGLE_CLOUD_PROJECT hoac dung --project-id."
            )

        self.base_url = config.resolve_endpoint()
        logger.info(
            "Khoi tao Vertex AI Provider, project: %s, region: %s.",
            config.project_id,
            config.region,
        )

    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        logger.debug("Goi Vertex AI de sinh anh dai dien (chua trien khai day du).")
        return None

    def check_quota(self) -> dict:
        return {
            "trang_thai": "chua_kiem_tra",
            "thong_bao": (
                "Kiem tra quota Vertex AI yeu cau xac thuc Service Account. "
                "Vui long su dung gcloud CLI de kiem tra."
            ),
        }

    def is_available(self) -> bool:
        return bool(self.config.project_id and self.config.api_key)


class OpenAIProvider(BaseAPIProvider):
    """Kết nối với OpenAI API (DALL-E cho sinh ảnh)."""

    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter

        if not config.api_key:
            raise APIUnavailableError(
                "API key chua duoc cau hinh cho OpenAI. "
                "Vui long thiet lap bien moi truong OPENAI_API_KEY hoac dung --api-key."
            )

        self.base_url = config.resolve_endpoint()
        logger.info("Khoi tao OpenAI Provider.")

    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        """Sinh ảnh đại diện sử dụng DALL-E."""
        logger.debug("Goi OpenAI DALL-E de sinh anh dai dien.")
        try:
            payload = {
                "model": "dall-e-3",
                "prompt": f"Simple passport photo portrait. {prompt}. "
                          "White background, professional, no text.",
                "n": 1,
                "size": "256x256",
                "response_format": "url",
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/images/generations",
                data=body,
                method="POST",
            )
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                image_url = result["data"][0]["url"]

            # Tải ảnh về từ URL
            with urllib.request.urlopen(image_url, timeout=30) as img_resp:
                img_data = img_resp.read()
            return Image.open(io.BytesIO(img_data)).convert("RGB")

        except Exception as loi:
            logger.warning("Khong the sinh anh tu OpenAI: %s", loi)
            return None

    def check_quota(self) -> dict:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models", method="GET"
            )
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return {"trang_thai": "kha_dung", "thong_bao": "Ket noi thanh cong."}
        except Exception as loi:
            return {"trang_thai": "loi", "thong_bao": str(loi)}
        return {"trang_thai": "khong_ro"}

    def is_available(self) -> bool:
        return bool(self.config.api_key)


class AvatarPlaceholderService:
    """
    Dịch vụ sinh ảnh đại diện placeholder không cần API trả phí.
    Hỗ trợ nhiều nguồn: RoboHash, DiceBear, hoặc sinh nội bộ.
    """

    SERVICES = {
        "robohash": "https://robohash.org/{seed}?size={w}x{h}&set=set5",
        "dicebear_pixel": "https://api.dicebear.com/7.x/pixel-art/png?seed={seed}&size={w}",
        "dicebear_lorelei": "https://api.dicebear.com/7.x/lorelei/png?seed={seed}&size={w}",
    }

    def __init__(self, service: str = "robohash"):
        self.service = service if service in self.SERVICES else "robohash"
        logger.debug("Khoi tao AvatarPlaceholderService voi dich vu: %s.", self.service)

    def fetch_avatar(self, seed: str, width: int = 200, height: int = 200) -> Optional[Image.Image]:
        """
        Tải ảnh đại diện placeholder từ dịch vụ trực tuyến.

        Tham số:
            seed: Chuỗi làm hạt giống để tạo avatar nhất quán.
            width: Chiều rộng ảnh (pixel).
            height: Chiều cao ảnh (pixel).

        Trả về:
            Đối tượng PIL Image hoặc None nếu thất bại.
        """
        url_template = self.SERVICES[self.service]
        url = url_template.format(
            seed=urllib.parse.quote(seed),
            w=width,
            h=height,
        )

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SyntheticDocGenerator/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                img_data = resp.read()
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as loi:
            logger.warning(
                "Khong the tai avatar tu %s (seed=%s): %s", self.service, seed, loi
            )
            return None

    def generate_local_avatar(self, seed: str, width: int = 200, height: int = 250) -> Image.Image:
        """
        Sinh ảnh đại diện đơn giản nội bộ khi không có kết nối mạng.
        Tạo ảnh hình học đơn giản với màu sắc dựa trên seed.

        Tham số:
            seed: Chuỗi hạt giống để đảm bảo tính nhất quán.
            width: Chiều rộng ảnh.
            height: Chiều cao ảnh.

        Trả về:
            Đối tượng PIL Image.
        """
        from PIL import ImageDraw

        # Tạo màu nền dựa trên seed
        seed_hash = hash(seed) % (256 ** 3)
        bg_r = (seed_hash >> 16) & 0xFF
        bg_g = (seed_hash >> 8) & 0xFF
        bg_b = seed_hash & 0xFF

        # Đảm bảo màu nền đủ sáng (ảnh chân dung thường có nền nhạt)
        bg_r = min(180 + (bg_r % 60), 235)
        bg_g = min(180 + (bg_g % 60), 235)
        bg_b = min(185 + (bg_b % 60), 240)

        img = Image.new("RGB", (width, height), color=(bg_r, bg_g, bg_b))
        draw = ImageDraw.Draw(img)

        # Vẽ hình người đơn giản
        skin_r = 210 + (seed_hash % 30)
        skin_g = 170 + (seed_hash % 40)
        skin_b = 140 + (seed_hash % 50)
        skin_color = (min(skin_r, 255), min(skin_g, 255), min(skin_b, 255))

        # Đầu
        head_cx, head_cy = width // 2, height // 3
        head_r = width // 5
        draw.ellipse(
            [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
            fill=skin_color,
        )

        # Thân
        shoulder_y = head_cy + head_r
        body_bottom = height
        draw.rectangle(
            [width // 4, shoulder_y, 3 * width // 4, body_bottom],
            fill=(80 + (seed_hash % 80), 80 + (seed_hash % 60), 100 + (seed_hash % 80)),
        )

        return img


class APIClient:
    """
    Lớp điều phối kết nối API chính.

    Tự động lựa chọn nhà cung cấp API phù hợp dựa trên cấu hình,
    xử lý lỗi và cung cấp giao diện thống nhất cho toàn bộ hệ thống.
    """

    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        """
        Khởi tạo APIClient và chọn provider phù hợp.

        Tham số:
            config: Cấu hình API.
            rate_limiter: Đối tượng kiểm soát tốc độ.
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self.provider = self._initialize_provider()
        self.avatar_placeholder = AvatarPlaceholderService(
            service=getattr(config, "avatar_placeholder_service", "robohash")
        )

        logger.info(
            "Khoi tao APIClient voi nen tang: %s.", config.platform
        )

    def _initialize_provider(self) -> BaseAPIProvider:
        """Khởi tạo nhà cung cấp API phù hợp với cấu hình."""
        platform = self.config.platform

        try:
            if platform == PLATFORM_GOOGLE_AI_STUDIO:
                return GoogleAIStudioProvider(self.config, self.rate_limiter)
            elif platform == PLATFORM_VERTEX_AI:
                return VertexAIProvider(self.config, self.rate_limiter)
            elif platform == PLATFORM_OPENAI:
                return OpenAIProvider(self.config, self.rate_limiter)
            else:
                logger.info(
                    "Nen tang API '%s': su dung che do khong API.", platform
                )
                return NoAPIProvider()
        except APIUnavailableError as loi:
            logger.warning(
                "Khong the khoi tao provider '%s': %s. Chuyen sang che do khong API.",
                platform,
                loi,
            )
            return NoAPIProvider()

    def get_avatar(
        self,
        seed: str,
        width: int = 200,
        height: int = 250,
        use_api: bool = False,
        gender: str = "unknown",
    ) -> Image.Image:
        """
        Lấy ảnh đại diện từ API hoặc dịch vụ placeholder.

        Ưu tiên theo thứ tự:
            1. API được cấu hình (nếu use_api=True và provider khả dụng).
            2. Dịch vụ placeholder trực tuyến (RoboHash, DiceBear).
            3. Ảnh đại diện sinh nội bộ (fallback cuối cùng).

        Tham số:
            seed: Chuỗi hạt giống để tạo avatar nhất quán.
            width: Chiều rộng ảnh.
            height: Chiều cao ảnh.
            use_api: Có sử dụng API AI để sinh ảnh không.
            gender: Giới tính ('male', 'female', 'unknown').

        Trả về:
            Đối tượng PIL Image.
        """
        # Thử sinh qua API nếu được yêu cầu
        if use_api and self.provider.is_available():
            prompt = f"A {gender if gender != 'unknown' else 'person'} passport photo portrait"
            try:
                api_image = self.rate_limiter.execute_with_retry(
                    self.provider.generate_avatar_image,
                    prompt,
                )
                if api_image:
                    logger.debug("Sinh anh dai dien thanh cong qua API.")
                    return api_image.resize((width, height), Image.Resampling.LANCZOS)
            except (RateLimitExceededError, APIError) as loi:
                logger.warning(
                    "Sinh anh qua API that bai, chuyen sang placeholder: %s", loi
                )

        # Thử dịch vụ placeholder trực tuyến
        avatar = self.avatar_placeholder.fetch_avatar(seed, width, height)
        if avatar:
            return avatar.resize((width, height), Image.Resampling.LANCZOS)

        # Fallback: sinh nội bộ
        logger.debug("Dung avatar sinh noi bo (fallback).")
        return self.avatar_placeholder.generate_local_avatar(seed, width, height)

    def check_api_status(self) -> dict:
        """
        Kiểm tra trạng thái và hạn mức của API hiện tại.

        Trả về:
            Từ điển thông tin trạng thái API.
        """
        return self.provider.check_quota()

    def is_api_available(self) -> bool:
        """Kiểm tra API có khả dụng không."""
        return self.provider.is_available()
        
    def generate_document_image(self, template_image: Image.Image, json_data: dict) -> Optional[Image.Image]:
        """Giao diện chung gọi API sinh ảnh tài liệu."""
        if not self.provider.is_available():
            logger.error("API khong kha dung de sinh anh.")
            return None
            
        return self.rate_limiter.execute_with_retry(
            getattr(self.provider, "generate_document_image", None),
            template_image,
            json_data
        )
    def generate_document(self, template_image: Image.Image, json_data: dict) -> Optional[Image.Image]:
        """
        Giao diện chung gọi API sinh ảnh tài liệu.
        Truyền dữ liệu xuống Provider tương ứng và áp dụng cơ chế thử lại.
        """
        if not self.provider.is_available():
            logger.error("API khong kha dung de sinh anh.")
            return None
            
        provider_method = getattr(self.provider, "generate_document_image", None)
        if not provider_method:
            logger.error("Provider hien tai chua ho tro sinh anh tai lieu tu JSON.")
            return None

        return self.rate_limiter.execute_with_retry(
            provider_method,
            template_image,
            json_data
        )
