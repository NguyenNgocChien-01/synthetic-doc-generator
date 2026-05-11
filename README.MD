# Synthetic Document Generator


`.env`:

```env
GOOGLE_API_KEY=...[Gemini API + Biling]
GOOGLE_CLOUD_PROJECT=...[ID_Project]

```


---

### Script batch 

```bash
# ./run.sh [số_mẫu] [delay_giây] [loại_tài_liệu]

./run.sh 10 15 passport
./run.sh 20 15 driver_license
```

**Tham số:**
- `số_mẫu` — bao nhiêu ảnh cần sinh (mặc định: 10)
- `delay_giây` — thời gian chờ giữa mỗi lần gọi API để tránh rate limit (mặc định: 8, khuyến nghị: 15)
- `loại_tài_liệu` — `passport` hoặc `driver_license`... (mặc định: passport)

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
│   │   |── template.jpg ← ảnh template bằng lái
|   |   └── base.json ← json genarate
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
