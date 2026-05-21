import json
import urllib.request
import urllib.error
import base64
import os
import google.auth
import google.auth.transport.requests

PROJECT_ID = "first-orc-chien"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash-image"

INPUT_PATH = r"templates\aus_medicare_card\blue\template.png"  # ảnh đầu vào
OUTPUT_PATH = "test_gemini_output.png"

credentials, _ = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
ACCESS_TOKEN = credentials.token

url = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
)

# Đọc ảnh input
with open(INPUT_PATH, "rb") as f:
    input_b64 = base64.b64encode(f.read()).decode()
mime = "image/png" if INPUT_PATH.endswith(".png") else "image/jpeg"

payload = {
    "contents": [{
        "role": "user",
        "parts": [
            # ✅ Ảnh trước
            {"inlineData": {"mimeType": mime, "data": input_b64}},
            # ✅ Text sau
            {"text": """

Prompt to API:
 Task: Generate a photorealistic aus_medicare_card by replacing specific data fields on the provided template.

Replace ONLY text fields (medicare_card_number, cardholders, expiry_date). Preserve background, watermark, font style, 
color, and layout exactly. Cardholders: multiline, evenly spaced rows, columns aligned.

## BASE_JSON (OLD DATA TO ERASE):
Locate these values on the template and erase them seamlessly:
{
  "document_type": "AUS_MEDICARE_CARD",
  "card_type": "interim card",
  "medicare_card_number": "1234 56789 1",
  "cardholders": "1[use tab]JOHN[use tab]A[use tab]CITIZEN[NextLine]",
  "expiry_date": "09/09/2009"
}

## TARGET_DATA (NEW DATA TO INJECT):
Render EXACTLY these values into the newly erased spatial positions:
{
  "document_type": "AUS_MEDICARE_CARD",
  "card_type": "interim card",
  "medicare_card_number": "5942 28316 3",
  "cardholders": "1[use tab]ERIC[use tab][use tab]STEELE[NextLine]\n2[use tab]SANDY[use tab][use 
tab]ANDERSON[NextLine]\n3[use tab]ASHLEY[use tab][use tab]SANTIAGO[NextLine]\n4[use tab]SAVANNAH[use tab][use 
tab]HILL[NextLine]\n5[use tab]SHANE[use tab][use tab]HUDSON[NextLine]",
  "expiry_date": "17/06/2029"
}

## PHOTO REPLACEMENT
CRITICAL INSTRUCTION: DO NOT add any face.

## FINAL OUTPUT REQUIREMENT:
Return ONLY the modified photorealistic image. No text explanations, no borders, no layout shifts. """
           
            }
        ]
    }],
    "generationConfig": {
        "responseModalities": ["IMAGE"]  # chỉ trả về ảnh
    }
}

req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")

print(f"Gửi IMG+TEXT → IMG tới {MODEL}...")

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())

        img_b64 = None
        for part in res.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                img_b64 = part["inlineData"]["data"]
                break
            if "text" in part:
                print("Model:", part["text"])

        if img_b64:
            with open(OUTPUT_PATH, "wb") as f:
                f.write(base64.b64decode(img_b64))
            print(f"[OK] Lưu tại: {OUTPUT_PATH}")
        else:
            print("[FAIL] Không có ảnh")
            print(json.dumps(res, indent=2))

except urllib.error.HTTPError as e:
    print(f"[HTTP {e.code}]", e.read().decode())
except Exception as e:
    print(f"[ERR] {e}")