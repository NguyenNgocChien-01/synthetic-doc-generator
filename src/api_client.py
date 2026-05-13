
"""
Module kết nối với Vertex AI.
"""

import io
import json
import logging
import time
import base64
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional
from PIL import ImageOps, Image
# from rich.prompt import result

from .config import APIConfig, PLATFORM_VERTEX_AI, PROMPT_TEMPLATES
from .rate_limiter import RateLimiter, RateLimitExceededError

logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class APIUnavailableError(APIError):
    pass

class BaseAPIProvider(ABC):
    @abstractmethod
    def generate_avatar_image(self, prompt: str) -> Optional[Image.Image]:
        pass

    @abstractmethod
    def check_quota(self) -> dict:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class VertexAIProvider(BaseAPIProvider):
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


    def _get_access_token(self) -> str:
        # 1. Thử xác thực qua môi trường Local (Application Default Credentials)
        try:
            import google.auth
            from google.auth.transport.requests import Request as GoogleRequest
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(GoogleRequest())
            if creds.token:
                return creds.token
        except Exception as local_auth_err:
            logger.debug("Khong the xac thuc qua google-auth (Local): %s", local_auth_err)

        # 2. Thử xác thực qua Metadata Server (Google Cloud VM)
        url = (
            "http://metadata.google.internal/computeMetadata/v1"
            "/instance/service-accounts/default/token"
        )
        req = urllib.request.Request(url)
        req.add_header("Metadata-Flavor", "Google")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["access_token"]
        except Exception as metadata_err:
            raise APIError(
                f"Khong the lay access token tu ca hai moi truong. Lỗi Metadata: {metadata_err}"
            ) from metadata_err

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

    def generate_document_image(
        self, template_image: Image.Image, json_data: dict
    ) -> Optional[Image.Image]:
        image_model = self.config.image_model or "gemini-2.5-flash-image"
        logger.debug("Goi Vertex AI Gemini voi mo hinh: %s", image_model)

        template_image = ImageOps.exif_transpose(template_image)
        orig_w, orig_h = template_image.size
        orig_is_portrait = orig_h > orig_w
        logger.debug(
            "Template goc: %dx%d, is_portrait=%s", orig_w, orig_h, orig_is_portrait
        )

        return self._generate_single_page(
            template_image, json_data, image_model, orig_is_portrait
        )

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
    
            faces_sorted = sorted(faces, key=lambda f: f[1])
            x_s, y_s, w_s, h_s = faces_sorted[0]  
            x_l, y_l, w_l, h_l = faces_sorted[1]  
    
            logger.debug(
                "Slot nho (top): x=%d y=%d w=%d h=%d | Slot lon (bot): x=%d y=%d w=%d h=%d",
                x_s, y_s, w_s, h_s, x_l, y_l, w_l, h_l,
            )
    
            pad_x = int(w_l * 0.30)
            pad_y = int(h_l * 0.35)
            crop_x1 = max(0, x_l - pad_x)
            crop_y1 = max(0, y_l - pad_y)
            crop_x2 = min(W, x_l + w_l + pad_x)
            crop_y2 = min(H, y_l + h_l + pad_y)
            avatar_large = output_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
            pad_x2 = int(w_s * 0.30)
            pad_y2 = int(h_s * 0.35)
            paste_x1 = max(0, x_s - pad_x2)
            paste_y1 = max(0, y_s - pad_y2)
            paste_x2 = min(W, x_s + w_s + pad_x2)
            paste_y2 = min(H, y_s + h_s + pad_y2)
            paste_w = paste_x2 - paste_x1
            paste_h = paste_y2 - paste_y1
    
            avatar_resized = avatar_large.resize((paste_w, paste_h), Image.LANCZOS)
    
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
        MAX_SIDE = 1536
        w, h = page_image.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            page_image = page_image.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )
            logger.debug("Resize template: %dx%d", page_image.width, page_image.height)

        buffered = io.BytesIO()
        page_image.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Lấy loại tài liệu và cấu hình prompt
        doc_type_raw = json_data.get("doc_type", "document")
        doc_type_key = doc_type_raw.lower().replace(" ", "_")
        
        prompt_config = PROMPT_TEMPLATES.get(doc_type_key, PROMPT_TEMPLATES["default"])
        date_format = prompt_config["date_format"]
        photo_instructions = prompt_config["photo_instructions"]

        # Xử lý các trường dữ liệu
        fields = {k: v for k, v in json_data.items() if k != "doc_type"}
        gender_raw = str(fields.get('sex', fields.get('gender', 'M'))).upper()
        target_gender = 'female' if gender_raw in ('F', 'FEMALE') else 'male'
        target_dob = fields.get('date_of_birth', fields.get('dob', ''))

        prompt_text = (
            f"This is a {doc_type_raw.replace('_', ' ')} document template image. "
            "Your task: generate a NEW photorealistic version of this document. "
            "STRICT RULES:\n"
            "1. LAYOUT: Keep the EXACT same layout, colors, fonts, logos, borders, watermarks, "
            "background patterns and structure as the template. Do not move or remove any element.\n"
            "2. TEXT: Replace ALL text fields with the following data EXACTLY as provided. "
            f"{date_format} "
            "Never alter, invent or distort any value:\n"
            f"{json.dumps(fields, ensure_ascii=False, indent=2)}\n"
            "3. PHOTOS: This document contains portrait photo slots. "
            "Generate ONE new fictional human face that does NOT resemble anyone in the template. "
            f"The face MUST match: gender={target_gender}, approximate age based on date_of_birth={target_dob}. "
            f"{photo_instructions}\n"
            "4. OUTPUT: One single photorealistic document image, "
            "same dimensions and orientation as the template."
        )
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
                "responseModalities": ["IMAGE"],
            },
        }

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

        try:
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                raw_data = None
                if "inlineData" in part:
                    raw_data = base64.b64decode(part["inlineData"]["data"])
                elif "inline_data" in part:
                    raw_data = base64.b64decode(part["inline_data"]["data"])

                if raw_data:
                    # 1. Load ảnh từ dữ liệu nhị phân
                    img = Image.open(io.BytesIO(raw_data))
                    
                    # 2. Xử lý Exif và hệ màu
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    
                    # --- SỬA TẠI ĐÂY: Xử lý xoay chiều và ép kích thước ---
                    # Lưu kích thước của ảnh mẫu hiện tại (page_image) để so sánh
                    orig_input_w, orig_input_h = page_image.size
                    
                    # A. Xoay chiều nếu cần (giữ expand=True để không mất pixel)
                    out_is_portrait = img.height > img.width
                    if is_portrait and not out_is_portrait:
                        img = img.rotate(90, expand=True)
                        logger.debug("Xoay output +90 ve portrait.")
                    elif not is_portrait and out_is_portrait:
                        img = img.rotate(-90, expand=True)
                        logger.debug("Xoay output -90 ve landscape.")

                    # B. QUAN TRỌNG NHẤT: Ép kích thước về khớp 100% ảnh gốc
                    # Việc này xử lý trường hợp Gemini trả về ảnh vuông hoặc sai aspect ratio
                    if img.size != (orig_input_w, orig_input_h):
                        logger.debug(
                            "Force resize output (%dx%d) ve khop template (%dx%d).",
                            img.width, img.height, orig_input_w, orig_input_h
                        )
                        img = img.resize((orig_input_w, orig_input_h), Image.Resampling.LANCZOS)
                    # --- KẾT THÚC ĐOẠN SỬA ---

                    # 3. Chạy hàm fix avatar (nếu có OpenCV)
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

class AvatarPlaceholderService:
    SERVICES = {
        "robohash": "https://robohash.org/{seed}?size={w}x{h}&set=set5",
        "dicebear_pixel": "https://api.dicebear.com/7.x/pixel-art/png?seed={seed}&size={w}",
        "dicebear_lorelei": "https://api.dicebear.com/7.x/lorelei/png?seed={seed}&size={w}",
    }

    def __init__(self, service: str = "robohash"):
        self.service = service if service in self.SERVICES else "robohash"
        logger.debug("Khoi tao AvatarPlaceholderService voi dich vu: %s.", self.service)

    def fetch_avatar(self, seed: str, width: int = 200, height: int = 200) -> Optional[Image.Image]:
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
        from PIL import ImageDraw

        seed_hash = hash(seed) % (256 ** 3)
        bg_r = (seed_hash >> 16) & 0xFF
        bg_g = (seed_hash >> 8) & 0xFF
        bg_b = seed_hash & 0xFF

        bg_r = min(180 + (bg_r % 60), 235)
        bg_g = min(180 + (bg_g % 60), 235)
        bg_b = min(185 + (bg_b % 60), 240)

        img = Image.new("RGB", (width, height), color=(bg_r, bg_g, bg_b))
        draw = ImageDraw.Draw(img)

        skin_r = 210 + (seed_hash % 30)
        skin_g = 170 + (seed_hash % 40)
        skin_b = 140 + (seed_hash % 50)
        skin_color = (min(skin_r, 255), min(skin_g, 255), min(skin_b, 255))

        head_cx, head_cy = width // 2, height // 3
        head_r = width // 5
        draw.ellipse(
            [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
            fill=skin_color,
        )

        shoulder_y = head_cy + head_r
        body_bottom = height
        draw.rectangle(
            [width // 4, shoulder_y, 3 * width // 4, body_bottom],
            fill=(80 + (seed_hash % 80), 80 + (seed_hash % 60), 100 + (seed_hash % 80)),
        )

        return img

class APIClient:
    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter
        self.provider = VertexAIProvider(self.config, self.rate_limiter)
        self.avatar_placeholder = AvatarPlaceholderService(
            service=getattr(config, "avatar_placeholder_service", "robohash")
        )

        logger.info(
            "Khoi tao APIClient voi nen tang: %s.", config.platform
        )

    def get_avatar(
        self,
        seed: str,
        width: int = 200,
        height: int = 250,
        use_api: bool = False,
        gender: str = "unknown",
    ) -> Image.Image:
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

        avatar = self.avatar_placeholder.fetch_avatar(seed, width, height)
        if avatar:
            return avatar.resize((width, height), Image.Resampling.LANCZOS)

        logger.debug("Dung avatar sinh noi bo (fallback).")
        return self.avatar_placeholder.generate_local_avatar(seed, width, height)

    def check_api_status(self) -> dict:
        return self.provider.check_quota()

    def is_api_available(self) -> bool:
        return self.provider.is_available()
        
    def generate_document_image(self, template_image: Image.Image, json_data: dict) -> Optional[Image.Image]:
        if not self.provider.is_available():
            logger.error("API khong kha dung de sinh anh.")
            return None
            
        return self.rate_limiter.execute_with_retry(
            getattr(self.provider, "generate_document_image", None),
            template_image,
            json_data
        )

    def generate_document(self, template_image: Image.Image, json_data: dict) -> Optional[Image.Image]:
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