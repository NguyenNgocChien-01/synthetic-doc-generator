# main.py
import argparse
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import Config, SUPPORTED_DOC_TYPES
from src.rate_limiter import RateLimiter
from src.api_client import APIClient
from src.image_processor import ImageProcessor
from src.storage import StorageManager
from src.progress_monitor import create_progress_monitor
from src.generator import DataGenerator

def setup_logging(debug_mode: bool):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main():
    parser = argparse.ArgumentParser(description="Cong cu sinh du lieu OCR tong hop.")
    parser.add_argument("--type", choices=SUPPORTED_DOC_TYPES, required=True, help="Loai tai lieu (vi du: passport)")
    parser.add_argument("--count", type=int, default=1, help="So luong mau can sinh")
    parser.add_argument("--workers", type=int, default=1, help="So luong luong xu ly song song")
    parser.add_argument("--debug", action="store_true", help="Bat che do debug")
    parser.add_argument("--avatar-api", action="store_true", help="Su dung API de sinh anh dai dien")
    parser.add_argument("--image-model", default="imagen-3.0-generate-001", help="Mo hinh sinh anh")
    parser.add_argument("--project-id", help="GCP Project ID su dung cho Vertex AI")
    parser.add_argument("--region", default="us-central1", help="GCP Region su dung cho Vertex AI")
    parser.add_argument("--state", default=None, help="Bang (vd: vic, nsw, act, nt...)")
    
    args = parser.parse_args()
    args.state = args.state.lower() if args.state else None

    config = Config.from_cli_args(args)
    setup_logging(config.debug_mode)

    rate_limiter = RateLimiter(requests_per_minute=300)
    api_client = APIClient(config.api, rate_limiter)
    storage_manager = StorageManager(dataset_dir=config.storage.dataset_dir)
    image_processor = ImageProcessor(templates_dir=config.storage.templates_dir, config=config.image)

    generator = DataGenerator(
        config=config,
        api_client=api_client,
        storage_manager=storage_manager,
        image_processor=image_processor,
        rate_limiter=rate_limiter
    )

    monitor = create_progress_monitor(args.count, args.type, prefer_rich=True)
    monitor.start()

    def progress_callback(result):
        try:
            monitor.update(
                success=result.success,
                sample_id=result.sample_id,
                error_message=result.error_message
            )
        except Exception:
            pass

    generator.generate_batch(
        doc_type=args.type,
        count=args.count,
        progress_callback=progress_callback,
        num_workers=config.num_workers,
        state=args.state,
    )

    monitor.finish()

if __name__ == "__main__":
    main()