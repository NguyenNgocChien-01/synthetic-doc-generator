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
from PIL import ImageOps

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
            
    def generate_document_image(
        self, template_image: Image.Image, json_data: dict
    ) -> Optional[Image.Image]:
        image_model = self.config.image_model or "gemini-2.5-flash-image"
        logger.debug("Goi Vertex AI Gemini voi mo hinh: %s", image_model)
    
        # Fix EXIF orientation của template trước khi gửi lên API
        template_image = ImageOps.exif_transpose(template_image)
    
        # Lưu kích thước GỐC sau khi đã fix EXIF
        orig_w, orig_h = template_image.size
        orig_is_portrait = orig_h >= orig_w
    
        # Chỉ resize nếu quá lớn, KHÔNG xoay
        MAX_SIDE = 1536
        w, h = orig_w, orig_h
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            template_image = template_image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
    
        buffered = io.BytesIO()
        template_image.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
 
        # 2. Xây dựng prompt
        doc_type = json_data.get("doc_type", "driver license")
        fields = {k: v for k, v in json_data.items() if k != "doc_type"}
        field_text = json.dumps(fields, ensure_ascii=False, indent=2)
        prompt_text = (
            f"This is a {doc_type} document template. "
            "Generate a new version of this document image with the same layout, "
            "background, logo, borders and structure, but fill in the text fields "
            "with the following data:\n"
            f"{field_text}\n"
            "Keep all visual elements identical. Only change the text content."
        )
 
        # 3. Payload theo chuẩn Gemini AI Studio
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {
                            "text": prompt_text
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }
 
        # 4. Gửi request
        url = (
            f"{self.base_url}/models/{model_name}"
            f":generateContent?key={self.config.api_key}"
        )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
 
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as loi:
            error_body = loi.read().decode("utf-8", errors="replace")
            logger.error("Loi HTTP %s tu AI Studio: %s", loi.code, error_body[:300])
            return None
        except Exception as loi:
            logger.error("Loi API: %s", loi)
            return None
 
        # 5. Trích xuất ảnh từ response Gemini
        try:
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                raw_data = None
                if "inlineData" in part:
                    raw_data = base64.b64decode(part["inlineData"]["data"])
                elif "inline_data" in part:
                    raw_data = base64.b64decode(part["inline_data"]["data"])
        
                if raw_data:
                    img = Image.open(io.BytesIO(raw_data))
        
                    # Fix EXIF orientation trước khi convert
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
        
                    # Nếu chiều output khác chiều template gốc → xoay lại cho khớp
                    orig_is_portrait = h >= w
                    out_is_portrait = img.height >= img.width
                    if orig_is_portrait != out_is_portrait:
                        img = img.rotate(90, expand=True)
                        logger.debug("Xoay anh output cho khop chieu voi template goc.")
        
                    logger.info("Sinh anh tai lieu thanh cong qua Vertex AI Gemini.")
                    return img
        
            logger.error(
                "Gemini tra ve response nhung khong co anh. Parts: %s",
                str(parts)[:300]
            )
            return None
        except (KeyError, IndexError) as loi:
            logger.error(
                "Khong the parse response: %s | Response: %s",
                loi, json.dumps(result)[:300]
            )
            return None
class VertexAIProvider(BaseAPIProvider):
    """Kết nối với Google Vertex AI — dùng gemini-2.5-flash-image để sinh ảnh tài liệu."""

    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter

        if not config.project_id:
            raise APIUnavailableError(
                "Project ID chua duoc cau hinh cho Vertex AI. "
                "Vui long thiet lap bien moi truong GOOGLE_CLOUD_PROJECT hoac dung --project-id."
            )

        self.region = config.region or "us-central1"
        self.project_id = config.project_id
        self.base_url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1"
            f"/projects/{self.project_id}/locations/{self.region}"
            f"/publishers/google/models"
        )

        logger.info(
            "Khoi tao Vertex AI Provider, project: %s, region: %s.",
            config.project_id,
            self.region,
        )

    # ------------------------------------------------------------------
    # Lấy access token từ metadata server
    # ------------------------------------------------------------------
    def _get_access_token(self) -> str:
        url = (
            "http://metadata.google.internal/computeMetadata/v1"
            "/instance/service-accounts/default/token"
        )
        req = urllib.request.Request(url)
        req.add_header("Metadata-Flavor", "Google")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["access_token"]
        except Exception as loi:
            raise APIError(f"Khong the lay access token: {loi}") from loi

    # ------------------------------------------------------------------
    # BaseAPIProvider interface
    # ------------------------------------------------------------------
    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        logger.debug("VertexAI: generate_avatar_image chua duoc trien khai.")
        return None

    def check_quota(self) -> dict:
        try:
            self._get_access_token()
            return {
                "trang_thai": "kha_dung",
                "thong_bao": f"Ket noi thanh cong toi Vertex AI (project: {self.project_id}).",
            }
        except APIError as loi:
            return {"trang_thai": "loi", "thong_bao": str(loi)}

    def is_available(self) -> bool:
        return bool(self.project_id)

    # ------------------------------------------------------------------
    # Sinh ảnh tài liệu — entry point chính
    # ------------------------------------------------------------------
    def generate_document_image(
        self, template_image: Image.Image, json_data: dict
    ) -> Optional[Image.Image]:
        image_model = self.config.image_model or "gemini-2.5-flash-image"
        logger.debug("Goi Vertex AI Gemini voi mo hinh: %s", image_model)

        # Bước 1: Fix EXIF orientation của template trước khi làm gì
        template_image = ImageOps.exif_transpose(template_image)

        # Bước 2: Lưu kích thước GỐC sau khi đã fix EXIF
        orig_w, orig_h = template_image.size
        orig_is_portrait = orig_h > orig_w
        logger.debug(
            "Template goc: %dx%d, is_portrait=%s", orig_w, orig_h, orig_is_portrait
        )

        return self._generate_single_page(
            template_image, json_data, image_model, orig_is_portrait
        )

    # ------------------------------------------------------------------
    # Xử lý 1 trang: resize → gửi API → fix chiều output
        # ------------------------------------------------------------------
    def _fix_avatar_consistency(self, output_img: Image.Image) -> Image.Image:
        try:
            import cv2
            import numpy as np
            from PIL import ImageDraw, ImageFilter
    
            W, H = output_img.size
            img_cv = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)
    
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(int(W * 0.04), int(H * 0.04)),
            )
    
            if len(faces) != 2:
                logger.debug("Phat hien %d khuon mat, bo qua fix avatar.", len(faces))
                return output_img
    
            # Sắp xếp theo y: faces[0] = trên (nhỏ/top-right), faces[1] = dưới (lớn/bottom-left)
            faces_sorted = sorted(faces, key=lambda f: f[1])
            x_s, y_s, w_s, h_s = faces_sorted[0]  # slot nhỏ — trên
            x_l, y_l, w_l, h_l = faces_sorted[1]  # slot lớn — dưới
    
            logger.debug(
                "Slot nho (top): x=%d y=%d w=%d h=%d | Slot lon (bot): x=%d y=%d w=%d h=%d",
                x_s, y_s, w_s, h_s, x_l, y_l, w_l, h_l,
            )
    
            # Crop avatar lớn (bottom) với padding
            pad_x = int(w_l * 0.30)
            pad_y = int(h_l * 0.35)
            crop_x1 = max(0, x_l - pad_x)
            crop_y1 = max(0, y_l - pad_y)
            crop_x2 = min(W, x_l + w_l + pad_x)
            crop_y2 = min(H, y_l + h_l + pad_y)
            avatar_large = output_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
            # Vùng paste vào slot nhỏ (top) với padding tương tự
            pad_x2 = int(w_s * 0.30)
            pad_y2 = int(h_s * 0.35)
            paste_x1 = max(0, x_s - pad_x2)
            paste_y1 = max(0, y_s - pad_y2)
            paste_x2 = min(W, x_s + w_s + pad_x2)
            paste_y2 = min(H, y_s + h_s + pad_y2)
            paste_w = paste_x2 - paste_x1
            paste_h = paste_y2 - paste_y1
    
            avatar_resized = avatar_large.resize((paste_w, paste_h), Image.LANCZOS)
    
            # Blend biên mềm bằng rectangle mask
            mask_pil = Image.new("L", (paste_w, paste_h), 0)
            draw = ImageDraw.Draw(mask_pil)
            mx = int(paste_w * 0.06)
            my = int(paste_h * 0.06)
            draw.rectangle((mx, my, paste_w - mx, paste_h - my), fill=255)
            mask_pil = mask_pil.filter(
                ImageFilter.GaussianBlur(radius=min(paste_w, paste_h) * 0.05)
            )
    
            result = output_img.copy()
            bg_region = result.crop((paste_x1, paste_y1, paste_x2, paste_y2))
            blended = Image.composite(avatar_resized, bg_region, mask_pil)
            result.paste(blended, (paste_x1, paste_y1))
    
            logger.info(
                "Fix avatar: copy slot duoi -> slot tren tai (%d,%d)", paste_x1, paste_y1
            )
            return result
    
        except ImportError:
            logger.warning("opencv-python chua duoc cai, bo qua fix avatar.")
            return output_img
        except Exception as loi:
            logger.error("Loi khi fix avatar: %s", loi)
            return output_img
                
    def _generate_single_page(
        self,
        page_image: Image.Image,
        json_data: dict,
        image_model: str,
        is_portrait: bool,
    ) -> Optional[Image.Image]:
        # Resize nếu quá lớn, KHÔNG xoay
        MAX_SIDE = 1536
        w, h = page_image.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            page_image = page_image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
            logger.debug("Resize template: %dx%d", page_image.width, page_image.height)

        # Encode ảnh
        buffered = io.BytesIO()
        page_image.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Xây dựng prompt
        doc_type = json_data.get("doc_type", "driver license")
        fields = {k: v for k, v in json_data.items() if k != "doc_type"}
        prompt_text = (
            f"This is a {doc_type} document template image. "
            "Your task: generate a NEW photorealistic version of this document. "
            "STRICT RULES:\n"
            "1. LAYOUT: Keep the EXACT same layout, colors, fonts, logos, borders, watermarks, "
            "background patterns and structure as the template. Do not move or remove any element.\n"
            "2. TEXT: Replace ALL text fields with the following data EXACTLY as provided. "
            "ALL dates MUST use format 'DD MMM YYYY' (example: 23 JAN 2025). "
            "Never alter, invent or distort any value:\n"
            f"{json.dumps(fields, ensure_ascii=False, indent=2)}\n"
            "3. PHOTOS: This document contains portrait photo slots. "
            "Generate ONE new fictional human face that does NOT resemble anyone in the template. "
            f"The face MUST match: gender={'female' if str(fields.get('sex', fields.get('gender', 'M'))).upper() in ('F', 'FEMALE') else 'male'}, "
            f"approximate age based on date_of_birth={fields.get('date_of_birth', fields.get('dob', ''))}. "
            "Place this new face ONLY in the LARGEST portrait photo slot. "
            "For any other smaller portrait slots, leave them as they appear in the template.\n"
            "4. OUTPUT: One single photorealistic document image, "
            "same dimensions and orientation as the template. "
        )

        # Payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"text": prompt_text},
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }

        # Gửi request
        image_model_clean = image_model.replace("models/", "")
        url = f"{self.base_url}/{image_model_clean}:generateContent"
        logger.debug("Vertex AI endpoint: %s", url)

        try:
            token = self._get_access_token()
        except APIError as loi:
            logger.error("Khong the xac thuc voi Vertex AI: %s", loi)
            return None

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as loi:
            error_body = loi.read().decode("utf-8", errors="replace")
            logger.error("Loi HTTP %s tu Vertex AI: %s", loi.code, error_body[:500])
            return None
        except urllib.error.URLError as loi:
            logger.error("Loi ket noi toi Vertex AI: %s", loi.reason)
            return None
        except Exception as loi:
            logger.error("Loi khong xac dinh: %s", loi)
            return None

        # Trích xuất và fix chiều ảnh output
        try:
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                raw_data = None
                if "inlineData" in part:
                    raw_data = base64.b64decode(part["inlineData"]["data"])
                elif "inline_data" in part:
                    raw_data = base64.b64decode(part["inline_data"]["data"])

                if raw_data:
                    img = Image.open(io.BytesIO(raw_data))
                
                    # Fix EXIF
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                
                    # Fix chiều TRƯỚC
                    out_is_portrait = img.height > img.width
                    if is_portrait and not out_is_portrait:
                        img = img.rotate(90, expand=True)
                        logger.debug("Xoay output +90 ve portrait.")
                    elif not is_portrait and out_is_portrait:
                        img = img.rotate(-90, expand=True)
                        logger.debug("Xoay output -90 ve landscape.")
                
                    # Fix avatar SAU khi đã đúng chiều
                    img = self._fix_avatar_consistency(img)
                
                    logger.info("Sinh anh tai lieu thanh cong qua Vertex AI Gemini.")
                    return img

            logger.error(
                "Gemini tra ve response nhung khong co anh. Parts: %s",
                str(parts)[:300],
            )
            return None

        except (KeyError, IndexError) as loi:
            logger.error(
                "Khong the parse response: %s | Response: %s",
                loi, json.dumps(result)[:300],
            )
            return None
            
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
