# Synthetic Document Generator


`.env`:

```env
GOOGLE_API_KEY=...[Gemini API + Biling]
GOOGLE_CLOUD_PROJECT=...[ID_Project]

```
on Local
```
https://cloud.google.com/sdk/docs/install#windows
gcloud auth application-default login

# tạm thời k cần
<!-- gcloud compute ssh instance-20260506-083338 --zone=us-central1-b --project=first-orc-chien
cd /home/jupyter/synthetic-doc-generator
sudo chmod -R 777 dataset/ -->

```

---

### Script batch 

```bash
# ./run.sh [số_mẫu] [delay_giây] [loại_tài_liệu] [bang]

./run.sh 10 15 passport
./run.sh 20 15 driver_license
./run.sh 20 15 driver_license vic
```

**Tham số:**
- `số_mẫu` — bao nhiêu ảnh cần sinh (mặc định: 10)
- `delay_giây` — thời gian chờ giữa mỗi lần gọi API để tránh rate limit (mặc định: 8, khuyến nghị: 15)
- `loại_tài_liệu` — `passport` hoặc `driver_license`... (mặc định: passport)
- `bang` — (đang vic, có thể thêm cac thư mục template cho cac bang tương ứng).

### CLI 

```bash
python main.py \
    --type passport \
    --count 1 \
    --workers 1 \
    --project-id [ID-Project] \
    --image-model gemini-2.5-flash-image
```

---

## Cấu trúc thư mục

```
synthetic-doc-generator/
├── main.py
├── run.sh
├── templates/
│   ├── driver_license/
|   │   │─vic/
|   |       |── template.jpg ← ảnh template bằng lái
|   |       └── base.json ← json genarate
│   └── passport/
│       └── template.jpg ← ảnh template hộ chiếu
|          ...      
└── dataset/                   ← output sinh ra
    |── documents/
    |   ├── passport
    |   └── ...
    ├── labels/
        ├── passport
        └── ...
```
