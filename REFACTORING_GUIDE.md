# Synthetic Document Generator - Refactored Architecture

## Overview

This is a lightweight, **local-first** Python application for generating synthetic documents with realistic data and AI-rendered images. It focuses on three core pillars:

1. **Data Generation**: Fake data generation using Faker library
2. **Image Rendering**: Vertex AI Gemini/Imagen for realistic document images
3. **Local Storage**: Structured output to disk (JSON + image pairs)

## Project Structure

```
synthetic-doc-generator/
├── main.py                 # Entry point - local-first execution
├── requirements.txt        # Python dependencies
├── README.MD              # This file
├── run.sh                 # Shell script wrapper
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration & prompt templates
│   ├── api_client.py      # Vertex AI Gemini/Imagen API client
│   ├── faker_factory.py   # Fake data generation
│   ├── generator.py       # Main orchestration pipeline
│   ├── image_processor.py # Image augmentation & post-processing
│   ├── storage.py         # File I/O and directory management
│   ├── rate_limiter.py    # API call rate limiting
│   └── progress_monitor.py# Progress tracking (optional)
│
├── templates/             # Document templates (images)
│   ├── passport/
│   ├── driver_license/
│   └── medicare_card/
│
└── output/                # Generated documents (created at runtime)
    ├── passport/          # {sample_id}.json + {sample_id}.jpg
    ├── driver_license/
    └── medicare_card/
```

## Core Architecture

### 1. Config Module (`config.py`)

**Responsibility**: Store configuration, prompts, and settings.

**Key Components**:
- `APIConfig`: Vertex AI connection settings (project ID, region, models)
- `ImageConfig`: Augmentation parameters (rotation, brightness, blur)
- `StorageConfig`: Output paths and naming conventions
- `PROMPT_TEMPLATES`: Gemini system instructions and user prompts for each document type

**One-Shot Prompting**: Templates can include `base_json` coordinates to teach the AI document layout.

### 2. API Client Module (`api_client.py`)

**Responsibility**: Handle all Vertex AI API interactions.

**Class**: `VertexAIClient`

**Key Methods**:
```python
# Document rendering via Gemini vision
generate_document_image(template_image, json_data, doc_type, coordinates)
  ├─ _preprocess_image()     # Resize, ensure RGB
  ├─ _extract_context()      # Extract gender, DOB (NOT place_of_birth)
  ├─ _build_gemini_payload() # Construct API request
  ├─ _call_gemini_api()      # HTTP request to Gemini
  └─ _postprocess_image()    # Format conversion

# Avatar generation via Imagen
generate_avatar_image(prompt)  # Generate fictional face

# Health checks
check_quota()                  # Verify API connectivity
is_available()                 # Check configuration
```

**Authentication**: 
- Tries `google-auth` library first (recommended)
- Falls back to GCP metadata server
- Supports service accounts and user credentials

### 3. Faker Factory Module (`faker_factory.py`)

**Responsibility**: Generate realistic fake data.

**Key Features**:
- 40+ field types (names, dates, IDs, addresses, etc.)
- Australian locale support (ABN, postcode, phone)
- **BUG FIX**: `place_of_birth` does NOT trigger `dob` logic
  - Only explicit `["dob", "date_of_birth"]` keys trigger DOB generation
- **MRZ Generation**: Passports get exactly 44-character MRZ lines

**Supported Field Types**:
```python
FAKER_TYPE_HANDLERS = {
    "full_name", "first_name", "last_name", "gender",
    "date_of_birth", "nationality", "country",
    "passport_number", "license_number", "medicare_number",
    "issue_date", "expiry_date",
    "address", "city", "state", "postcode", "phone_number",
    "mrz_line1", "mrz_line2",
    ... 40+ more
}
```

### 4. Generator Module (`generator.py`)

**Responsibility**: Orchestrate the entire generation pipeline.

**Class**: `DataGenerator`

**Pipeline** (for each document):
```
1. Generate JSON Data
   └─ FakerFactory.generate_value() for each field

2. Load Template
   └─ ImageProcessor.load_template(doc_type, state)

3. Render Document
   └─ VertexAIClient.generate_document_image(template, json, doc_type)

4. Post-Process Image
   └─ ImageProcessor.post_process_image() [augmentation, format]

5. Save Pair
   └─ StorageManager.save_sample() [JSON + JPEG/PNG]
```

**Execution Modes**:
- **Serial**: Single-threaded (default)
- **Parallel**: ThreadPoolExecutor with N workers

**Error Handling**: 
- `GenerationResult` tracks success/failure for each document
- Individual errors don't stop the batch
- Comprehensive error messages and logging

### 5. Image Processor Module (`image_processor.py`)

**Responsibility**: Template management and image augmentation.

**Class**: `ImageProcessor`

**Key Methods**:
```python
load_template(doc_type, state)
  └─ Searches templates/{doc_type}/{state}/template.* (cached)

post_process_image(image)
  └─ Ensure RGB + apply augmentation

_apply_augmentation(image)
  ├─ Random rotation (up to 2° by default)
  ├─ Brightness adjustment (0.9x - 1.1x)
  ├─ Gaussian blur (15% probability)
  └─ Returns augmented image for training diversity

encode_image_to_bytes(image, format, quality)
  └─ Convert PIL Image to JPEG/PNG bytes
```

**Augmentation Settings** (configurable):
- `augmentation_rotation_max_degrees`: 2.0
- `augmentation_brightness_range`: (0.90, 1.10)
- `augmentation_blur_probability`: 0.15
- `augmentation_blur_radius`: 0.8
- Optional Gaussian noise (requires numpy)

### 6. Storage Module (`storage.py`)

**Responsibility**: Save and organize generated documents.

**Class**: `StorageManager`

**Output Structure**:
```
output/
  passport/
    ├── sample_001.json
    ├── sample_001.jpg
    ├── sample_002.json
    ├── sample_002.jpg
    ...
  driver_license/
    ├── nsw/
    │   ├── sample_001.json
    │   └── sample_001.jpg
    ...
```

**Key Methods**:
```python
ensure_directory(doc_type, state)      # Create output dirs
get_next_id(doc_type)                  # UUID or sequential ID
save_sample(doc_type, sample_id, image, json_data, state)
  └─ Atomic save: Both succeed or rollback
```

**Filename Matching**:
- `sample_id.json` and `sample_id.jpg` are guaranteed to correspond
- No guessing which JSON goes with which image

**Atomic Saves**:
- If image save succeeds but JSON fails, image is deleted
- Prevents orphaned files

### 7. Main Entry Point (`main.py`)

**Local-First Execution**:
```bash
# Simple usage
python main.py --type passport --count 10

# With options
python main.py --type driver_license \
  --count 50 \
  --workers 4 \
  --state nsw \
  --output-dir output \
  --project-id my-gcp-project
```

**CLI Arguments**:
```
--type            Document type (passport, driver_license, medicare_card) [REQUIRED]
--count           Number to generate (default: 1)
--workers         Parallel workers (default: 1)
--state           Document state (e.g., nsw, vic)
--output-dir      Output directory (default: output)
--project-id      GCP Project ID (or use GOOGLE_CLOUD_PROJECT env var)
--region          GCP region (default: us-central1)
--image-model     Gemini model (default: gemini-2.5-flash-image)
--debug           Enable debug logging
--dry-run         Skip actual generation
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ main.py - Parse CLI, Initialize Components             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ DataGenerator.generate_batch()                          │
│  ├─ Serial or Parallel Execution                        │
│  └─ For each sample_id:                                 │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────────┐      ┌──────────────────┐
    │ FakerFactory│      │ ImageProcessor   │
    │             │      │                  │
    │ Generate    │      │ Load Template    │
    │ JSON Data   │      │ (cached)         │
    └──────┬──────┘      └────────┬─────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
         ┌────────────────────────────────┐
         │ VertexAIClient.generate_        │
         │   document_image()              │
         │                                 │
         │ 1. Preprocess template          │
         │ 2. Build Gemini prompt          │
         │ 3. Call Gemini vision API       │
         │ 4. Parse image response         │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ ImageProcessor.post_process()  │
         │                                 │
         │ 1. Ensure RGB format            │
         │ 2. Apply augmentation           │
         │ 3. Return processed image       │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ StorageManager.save_sample()   │
         │                                 │
         │ 1. Create output directory      │
         │ 2. Save image as JPEG/PNG       │
         │ 3. Save JSON data               │
         │ 4. Both succeed or rollback     │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ GenerationResult                │
         │ ├─ success: bool                │
         │ ├─ sample_id: str               │
         │ ├─ image_path: str              │
         │ ├─ json_path: str               │
         │ └─ elapsed_seconds: float       │
         └────────────────────────────────┘
```

## Known Bug Fixes

### 1. Place of Birth vs Date of Birth
**Before**: Field named `place_of_birth` could accidentally trigger DOB generation logic
**After**: Only explicit `["dob", "date_of_birth"]` keys trigger DOB generation
**Verification**: Context extraction uses exact key matching

### 2. MRZ Line Length
**Before**: MRZ lines could have variable lengths (bug in passport rendering)
**After**: Both MRZ lines are guaranteed exactly 44 characters
**Verification**: Assertions check `len(mrz_line) == 44`

### 3. JSON + Image Pairing
**Before**: Mismatched image and JSON files could occur
**After**: Filenames match exactly (e.g., `doc_123.json` and `doc_123.jpg`)
**Mechanism**: StorageManager uses same `sample_id` for both

## Requirements

```
Pillow>=9.0.0          # Image manipulation
Faker>=18.0.0          # Fake data generation
google-auth>=2.0.0     # Google Cloud authentication
google-cloud-aiplatform # Vertex AI SDK (optional)
requests>=2.28.0       # HTTP client
tqdm>=4.65.0          # Progress bars
rich>=13.0.0          # Formatted output
python-dotenv>=1.0.0  # Environment variables
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud authentication**:
   ```bash
   # Option 1: Use default credentials
   gcloud auth application-default login
   
   # Option 2: Set environment variable
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ```

3. **Create template directory**:
   ```bash
   mkdir -p templates/{passport,driver_license,medicare_card}
   # Add template images: templates/passport/template.jpg, etc.
   ```

4. **Run**:
   ```bash
   python main.py --type passport --count 5
   ```

## Performance

- **Single document**: ~10-30 seconds (API latency)
- **Batch of 50 with 4 workers**: ~2-3 minutes
- **Rate limiting**: 300 requests/minute (configurable)
- **Image sizes**: Handled up to 1024x1024px
- **Output size**: ~500KB per document (JPEG + JSON)

## Error Handling

- **API errors**: Logged but don't stop batch
- **Network timeouts**: Configurable retry with exponential backoff
- **Storage failures**: Atomic saves with rollback
- **Malformed data**: Validated at each step

## Future Enhancements

- [ ] Async/await for faster batches
- [ ] Caching of generated documents
- [ ] Custom template selectors per document
- [ ] Metrics and monitoring (Prometheus)
- [ ] Distributed generation across machines
- [ ] Additional image augmentation strategies
- [ ] Support for more document types

## Troubleshooting

**"Project ID not configured"**:
```bash
export GOOGLE_CLOUD_PROJECT=my-project
# or
python main.py --project-id my-project --type passport
```

**"Template not found"**:
- Check `templates/{doc_type}/template.jpg` exists
- Verify file is readable and not corrupted

**"API quota exceeded"**:
- Reduce `--workers` parameter
- Increase wait time between batches
- Check GCP quotas in Cloud Console

**"Out of memory"**:
- Reduce `--workers` or `--count`
- Clear template cache between batches

## License

Internal Biwoco project.
