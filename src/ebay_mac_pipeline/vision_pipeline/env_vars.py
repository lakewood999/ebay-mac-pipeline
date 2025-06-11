import os

try:
    import dotenv

    # Load any environment variables from dotenv
    dotenv.load_dotenv()
except Exception as e:
    pass

OCR_BATCH_SIZE = os.getenv("OCR_BATCH_SIZE", 12)
OCR_REC_SIZE = os.getenv("OCR_REC_SIZE", 6)
SEGMENT_POINTS_PER_BATCH = os.getenv("SEGMENT_POINTS_PER_BATCH", 64)
SEGMENTATION_MODEL_PATH = os.getenv("SEGMENTATION_MODEL_PATH", "sam_vit_l_0b3195.pth")
SEGMENTATION_DEVICE = os.getenv("SEGMENTATION_DEVICE", "cuda")
TO_SEGMENT_DIR = os.getenv("TO_SEGMENT_DIR", "./to_segment")
TO_OCR_DIR = os.getenv("TO_OCR_DIR", "./to_ocr")
