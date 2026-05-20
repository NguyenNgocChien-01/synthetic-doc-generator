import json
import urllib.request
import urllib.error
import base64

# 1. Lấy API Key miễn phí từ: https://aistudio.google.com/app/apikey
API_KEY = "AIzaSyD7NyD0vcEyHPpi7scmBh5yqfHfecQkiNs"
MODEL = "imagen-3.0-generate-001" 

# Endpoint của Google AI Studio khác hoàn toàn Vertex AI
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:predict?key={API_KEY}"

# Payload theo chuẩn của Imagen trên AI Studio (dùng 'instances' thay vì 'contents')
payload = {
    "instances": [
        {
            "prompt": "A simple photorealistic image of a white coffee mug on a wooden table."
        }
    ],
    "parameters": {
        "sampleCount": 1,
        "aspectRatio": "16:9"
    }
}

req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
req.add_header("Content-Type", "application/json")

print("Đang gửi yêu cầu tạo ảnh tới Google AI Studio...")

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        
        predictions = res.get("predictions", [])
        if not predictions:
            print("[THẤT BẠI] Phản hồi không chứa 'predictions'.")
            print(json.dumps(res, indent=2))
        else:
            img_b64 = predictions[0].get("bytesBase64Encoded")
            if img_b64:
                print(f"[THÀNH CÔNG] Đã nhận được ảnh! Chiều dài base64: {len(img_b64)}")
                
                # Tuỳ chọn: Lưu ảnh ra file để kiểm tra trực quan
                with open("test_output.jpg", "wb") as fh:
                    fh.write(base64.b64decode(img_b64))
                print("Đã lưu ảnh thành công vào file 'test_output.jpg'")
            else:
                print("[CẢNH BÁO] Không tìm thấy dữ liệu ảnh.")
                print(json.dumps(res, indent=2))
                
except urllib.error.HTTPError as err:
    print(f"\n--- LỖI HTTP ---")
    print(f"Mã lỗi: {err.code}")
    print(f"Chi tiết: {err.read().decode('utf-8')}")
    if err.code == 429:
        print("\n=> GỢI Ý: Lỗi 429 là do vượt quá Rate Limit của gói Free. Hãy chờ khoảng 1 phút rồi thử lại.")
except Exception as e:
    print(f"\n--- LỖI HỆ THỐNG ---")
    print(f"Chi tiết: {e}")