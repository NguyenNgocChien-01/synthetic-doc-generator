# Synthetic Document Generator

## Setup

`.env`:

```env
GOOGLE_API_KEY=...[Gemini API + Billing]
GOOGLE_CLOUD_PROJECT=...[Project_ID]
```

**Local authentication:**
```bash
# Install SDK: https://cloud.google.com/sdk/docs/install#windows
gcloud auth application-default login
```

---

## Batch Script

```bash
# ./run.sh [count] [delay_seconds] [doc_type] [state]

./run.sh 10 15 passport
./run.sh 20 15 driver_license
./run.sh 20 15 driver_license vic
```

**Parameters:**
- `count` — number of images to generate (default: 10)
- `delay_seconds` — wait time between API calls to avoid rate limiting
- `doc_type` — `passport`, `driver_license`, `medicare_card`...
- `state` — state code for driver license: `vic`, `act`, `nsw`...

---

## CLI

```bash
python main.py \
    --type passport \
    --count 1 \
    --image-model gemini-2.5-flash-image
```

```bash
python main.py \
    --type driver_license \
    --state vic \
    --count 1 \
    --image-model gemini-2.5-flash-image
```

---

## Project Structure

```
synthetic-doc-generator/
├── main.py
├── run.sh
├── templates/
│   ├── driver_license/
│   │   └── vic/
│   │       ├── template.jpg
│   │       └── base.json
│   └── passport/
│       └── template.jpg...
└── dataset/
    ├── documents/
    │   ├── passport/
    │   └── driver_license/
    └── labels/
        ├── passport/
        └── driver_license/
```