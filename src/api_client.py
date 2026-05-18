"""
Module kết nối với Vertex AI.
"""

import io
import json
import logging
import base64
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional
from PIL import ImageOps, Image, ImageDraw, ImageFont

from .config import APIConfig, PROMPT_TEMPLATES
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
        try:
            import google.auth
            from google.auth.transport.requests import Request as GoogleRequest
            import requests

            session = requests.Session()
            session.timeout = (5, 15)
            google_request = GoogleRequest(session=session)

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google_request)

            if creds.token:
                logger.debug("Xac thuc thanh cong.")
                return creds.token

        except Exception as e:
            logger.debug("google-auth that bai: %s", e)

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
        except Exception as metadata_err:
            raise APIError(f"Khong the xac thuc: {metadata_err}") from metadata_err

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

    def _fix_avatar_consistency(self, output_img: Image.Image, doc_type_key: str) -> Image.Image:
        if "act" in doc_type_key:
            logger.debug("Bang lai ACT khong dung ghost face, bo qua fix_avatar.")
            return output_img

        try:
            import cv2
            import numpy as np
            from PIL import ImageFilter

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

            logger.info("Fix avatar: copy slot duoi -> slot tren tai (%d,%d)", paste_x1, paste_y1)
            return result

        except ImportError:
            logger.warning("opencv-python chua duoc cai, bo qua fix avatar.")
            return output_img
        except Exception as loi:
            logger.error("Loi khi fix avatar: %s", loi)
            return output_img

    def _preprocess_image(self, page_image: Image.Image) -> tuple[str, tuple[int, int]]:
        MAX_SIDE = 1024
        orig_size = page_image.size
        w, h = orig_size

        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            page_image = page_image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS,
            )
            logger.debug("Resize template: %dx%d", page_image.width, page_image.height)

        buffered = io.BytesIO()
        page_image.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_b64, orig_size

    def _extract_context(self, json_data: dict) -> tuple[str, dict]:
        doc_type_raw = json_data.get("document_type", json_data.get("doc_type", "document"))
        doc_type_key = doc_type_raw.lower().replace(" ", "_")

        if "driver_license" in doc_type_key:
            def find_issuing_state(data):
                if not isinstance(data, dict):
                    return ""
                state = data.get("issuing_state")
                if state:
                    return str(state).upper().strip()
                for value in data.values():
                    result = find_issuing_state(value)
                    if result:
                        return result
                return ""

            issuing_state = find_issuing_state(json_data)
            state_key = f"driver_license_{issuing_state.lower()}" if issuing_state else ""
            doc_type_key = state_key if state_key in PROMPT_TEMPLATES else "driver_license"

        prompt_config = PROMPT_TEMPLATES.get(doc_type_key, PROMPT_TEMPLATES.get("default", {}))

        front = json_data.get("front", {})
        holder = front.get("holder", {})

        gender = holder.get("sex") or front.get("sex") or json_data.get("sex")
        dob = holder.get("dob") or front.get("dob") or json_data.get("dob")

        if gender is None or dob is None:
            for value in json_data.values():
                if not isinstance(value, dict):
                    continue
                gender = gender or value.get("sex")
                dob = dob or value.get("dob")

        gender_str = str(gender).upper() if gender else "M"
        target_gender = "female" if gender_str in {"F", "FEMALE"} else "male"
        target_dob = str(dob) if dob else "unknown"

        info_rules = "\n".join(
            f"- {k.upper()}: {v}"
            for k, v in prompt_config.get("info", {}).items()
            if "mrz" not in k.lower()         
        )

        return doc_type_key, {
            "config": prompt_config,
            "target_gender": target_gender,
            "target_dob": target_dob,
            "info_rules": info_rules,
        }

    @staticmethod                               
    def _load_mrz_font(font_size: int, ocr_b_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
        """
        Load font monospace theo thứ tự ưu tiên:
        OCR-B → DejaVu Mono → Liberation Mono → FreeMono → Courier → PIL default
        """
        candidates = []
        if ocr_b_path:
            candidates.append(ocr_b_path)

        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            "/Library/Fonts/Courier New Bold.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "C:/Windows/Fonts/courbd.ttf",   # Courier Bold
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/consolab.ttf", # Consolas Bold
            "C:/Windows/Fonts/consola.ttf",
        ]

        for path in candidates:
            try:
                font = ImageFont.truetype(path, font_size)
                logger.debug("MRZ font loaded: %s @ %dpx", path, font_size)
                return font
            except (IOError, OSError):
                continue

        logger.warning("Khong load duoc font MRZ, dung PIL default bitmap.")
        return ImageFont.load_default()

    def _render_mrz_overlay(
        self,
        img: Image.Image,
        json_data: dict,
        doc_type_key: str,
        *,
        ocr_b_path: Optional[str] = None,
        bg_color: tuple[int, int, int] = (230, 225, 215),
        text_color: tuple[int, int, int] = (10, 10, 10),
        zone_ratio: float = 0.08,
        side_pad_ratio: float = 0.04,
    ) -> Image.Image:
        if "passport" not in doc_type_key and "td" not in doc_type_key:
            return img

        mrz1 = json_data.get("mrz_line1", "").strip().ljust(44, "<")[:44]
        mrz2 = json_data.get("mrz_line2", "").strip().ljust(44, "<")[:44]
        if not mrz1 and not mrz2:
            return img

        W, H = img.size
        zone_h   = int(H * zone_ratio)
        zone_top = H - zone_h
        side_pad = int(W * side_pad_ratio)

        usable_w  = W - 2 * side_pad
        max_chars = 44
        font_size = max(10, int((usable_w / max_chars) / 0.65))

        font = self._load_mrz_font(font_size, ocr_b_path)
        img  = img.copy()
        draw = ImageDraw.Draw(img)

        # Một lần duy nhất — xóa đúng vùng MRZ
        draw.rectangle([0, zone_top, W, H], fill=bg_color)

        line_h       = int(font_size * 1.35)
        total_text_h = 2 * line_h
        top_pad      = (zone_h - total_text_h) // 2
        y1 = zone_top + top_pad
        y2 = y1 + line_h

        for y, line in [(y1, mrz1), (y2, mrz2)]:
            if line:
                draw.text((side_pad, y), line, font=font, fill=text_color)

        logger.info(
            "MRZ rendered: zone_top=%d zone_h=%d font_size=%d",
            zone_top, zone_h, font_size,
        )
        return img

    def _build_payload(self, doc_type_key: str, json_data: dict, context: dict, img_b64: str) -> dict:
        config = context["config"]
        date_format = config.get("date_format", "")
        photo_instructions = config.get("photo_instructions", "")
        has_portrait_photo = config.get("has_portrait_photo", True)
        base_json = config.get("base_json")

        if has_portrait_photo:
            photo_section = (
                "## PHOTO SLOT\n"
                f"Generate ONE fictional human face. gender={context['target_gender']}, "
                f"approximate age from dob={context['target_dob']}.\n"
                f"{photo_instructions}\n"
                "IMPORTANT: Do NOT copy the face from the template image.\n"
            )
        else:
            photo_section = (
                "## PHOTO SLOT\n"
                "DO NOT generate any portrait, face, avatar, silhouette, or human figure.\n"
                f"{photo_instructions}\n"
            )

        if base_json:
            layout_section = (
                "## STEP 1 — LEARN LAYOUT FROM TEMPLATE\n"
                "Study the attached template image carefully.\n"
                "The JSON below maps each field to its EXACT position, font, size, "
                "alignment, rotation, and color in the template. Use it as a layout reference ONLY — "
                "do NOT copy these values into the output.\n\n"
                "CRITICAL: Reserve the bottom 9% of the image for MRZ zone. "
                "LAYOUT REFERENCE (do not output these values):\n"
                f"{json.dumps(base_json, ensure_ascii=False, indent=2)}\n\n"
                "## STEP 2 — FILL WITH NEW DATA\n"
                "Now replace every field using the TARGET DATA below. "
                "Keep the learned layout from Step 1 exactly — same positions, fonts, sizes.\n"
                f"DATE FORMAT RULES:\n{date_format}\n\n"
                "TARGET DATA (output exactly these values):\n"
                f"{json.dumps(json_data, ensure_ascii=False, indent=2)}\n"
            )
        else:
            layout_section = (
                "## FILL WITH NEW DATA\n"
                "Replace all text fields using the TARGET DATA below. "
                "Preserve the template layout exactly.\n"
                f"DATE FORMAT RULES:\n{date_format}\n\n"
                "TARGET DATA:\n"
                f"{json.dumps(json_data, ensure_ascii=False, indent=2)}\n"
            )

        info_rules = context.get("info_rules", "")
        specific_rules = f"## SPECIFIC RULES\n{info_rules}\n" if info_rules else ""

        mrz_section = ""    # MRZ do Pillow render sau — không cần AI

        prompt_text = (
            f"You are given a {doc_type_key} document template image.\n"
            "Generate ONE photorealistic version of this document with new data.\n\n"
            "## STRICT RULES\n"
            "1. LAYOUT: Keep EXACT same layout, colors, fonts, logos, watermarks, "
            "borders, background patterns. Do NOT move or remove any element.\n"
            "2. TEXT: Replace ALL text fields with TARGET DATA values EXACTLY as given. "
            "Never invent, alter, or omit any value.\n\n"
            f"{layout_section}\n"
            f"{photo_section}\n"
            f"{specific_rules}"
            f"{mrz_section}"
            "## OUTPUT\n"
            "Return ONE single photorealistic document image.\n"
            "Same dimensions and orientation as the template.\n"
            "Do NOT add borders, watermarks, or annotations not present in the template."
        )

        logger.debug("Prompt length: %d chars", len(prompt_text))
        print("Prompt to Vertex AI Gemini:\n", prompt_text[:10000], "...\n")

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                        {"text": prompt_text},
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            },
        }

    def _call_api(self, payload: dict, image_model: str) -> Optional[dict]:
        image_model_clean = image_model.replace("models/", "")
        url = f"{self.base_url}/{image_model_clean}:generateContent"
        logger.debug("Vertex AI endpoint: %s", url)

        try:
            token = self._get_access_token()
        except APIError as err:
            logger.error("Vertex AI auth error: %s", err)
            return None

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        logger.debug("Sending request to Vertex AI...")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            error_body = err.read().decode("utf-8", errors="replace")
            logger.error("Vertex AI HTTP %s: %s", err.code, error_body[:500])
        except urllib.error.URLError as err:
            logger.error("Vertex AI connection error: %s", err.reason)
        except TimeoutError:
            logger.error("Vertex AI timeout after 15s.")
        except Exception as err:
            logger.error("Unknown Vertex AI error: %s", err)

        return None

    def _postprocess_image(
        self,
        api_result: dict,
        orig_size: tuple[int, int],
        doc_type_key: str,
        is_portrait: bool,
        json_data: Optional[dict] = None,
    ) -> Optional[Image.Image]:
        try:
            parts = api_result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for part in parts:
                raw_data = None
                if "inlineData" in part:
                    raw_data = base64.b64decode(part["inlineData"]["data"])
                elif "inline_data" in part:
                    raw_data = base64.b64decode(part["inline_data"]["data"])

                if not raw_data:
                    continue

                img = Image.open(io.BytesIO(raw_data))
                img = ImageOps.exif_transpose(img).convert("RGB")

                out_is_portrait = img.height > img.width
                if is_portrait and not out_is_portrait:
                    img = img.rotate(90, expand=True)
                    logger.debug("Xoay output +90 ve portrait.")
                elif not is_portrait and out_is_portrait:
                    img = img.rotate(-90, expand=True)
                    logger.debug("Xoay output -90 ve landscape.")

                if img.size != orig_size:
                    logger.debug(
                        "Resize output (%dx%d) -> (%dx%d)",
                        img.width, img.height, orig_size[0], orig_size[1],
                    )
                    img = img.resize(orig_size, Image.Resampling.LANCZOS)

                img = self._fix_avatar_consistency(img, doc_type_key)

                if json_data:
                    img = self._render_mrz_overlay(
                        img,
                        json_data,
                        doc_type_key,
                        ocr_b_path=None,
                        bg_color=(230, 225, 215),
                        text_color=(10, 10, 10),
                        zone_ratio=0.11,
                        side_pad_ratio=0.04,
                    )

                logger.info("Document generated successfully.")
                return img

            logger.error("Gemini returned no image. Parts: %s", str(parts)[:300])
        except (KeyError, IndexError, Exception) as err:
            logger.error(
                "Failed parsing Gemini response: %s | %s",
                err, json.dumps(api_result)[:300],
            )

        return None

    def _generate_single_page(
        self,
        page_image: Image.Image,
        json_data: dict,
        image_model: str,
        is_portrait: bool,
    ) -> Optional[Image.Image]:
        img_b64, orig_size = self._preprocess_image(page_image)
        doc_type_key, context = self._extract_context(json_data)
        payload = self._build_payload(doc_type_key, json_data, context, img_b64)
        api_result = self._call_api(payload, image_model)

        if api_result:
            return self._postprocess_image(
                api_result, orig_size, doc_type_key, is_portrait, json_data
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
            fill=(
                80 + (seed_hash % 80),
                80 + (seed_hash % 60),
                100 + (seed_hash % 80),
            ),
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
        logger.info("Khoi tao APIClient voi nen tang: %s.", config.platform)

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
                logger.warning("Sinh anh qua API that bai, chuyen sang placeholder: %s", loi)

        avatar = self.avatar_placeholder.fetch_avatar(seed, width, height)
        if avatar:
            return avatar.resize((width, height), Image.Resampling.LANCZOS)

        logger.debug("Dung avatar sinh noi bo (fallback).")
        return self.avatar_placeholder.generate_local_avatar(seed, width, height)

    def check_api_status(self) -> dict:
        return self.provider.check_quota()

    def is_api_available(self) -> bool:
        return self.provider.is_available()

    def generate_document_image(
        self, template_image: Image.Image, json_data: dict
    ) -> Optional[Image.Image]:
        if not self.provider.is_available():
            logger.error("API khong kha dung de sinh anh.")
            return None

        return self.rate_limiter.execute_with_retry(
            getattr(self.provider, "generate_document_image", None),
            template_image,
            json_data,
        )

    def generate_document(
        self, template_image: Image.Image, json_data: dict
    ) -> Optional[Image.Image]:
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
            json_data,
        )