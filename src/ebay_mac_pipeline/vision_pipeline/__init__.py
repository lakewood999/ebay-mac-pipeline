import os, time, signal, cv2, json, argparse
from .helpers import apply_mask_output
from .env_vars import (
    SEGMENTATION_MODEL_PATH,
    SEGMENTATION_DEVICE,
    SEGMENT_POINTS_PER_BATCH,
    OCR_BATCH_SIZE,
    OCR_REC_SIZE,
    TO_SEGMENT_DIR,
    TO_OCR_DIR,
)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from paddleocr import PaddleOCR

def segment_images(files: list[str]):
    """
    Segments images using the SAM model and returns the segmentation results.

    :param files: List of image file names/paths to segment. Image type doesn't
    matter for this function. Files are assumed to exist in the TO_SEGMENT_DIR
    environment variable
    """
    try:
        print("Loading segmentation model...")
        sam = sam_model_registry["vit_l"](
            checkpoint=SEGMENTATION_MODEL_PATH,
        )
        sam.to(SEGMENTATION_DEVICE)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            points_per_batch=SEGMENT_POINTS_PER_BATCH,
            output_mode="coco_rle",
        )
        print("Segmentation model loaded")
        results = []
        for file_num in range(len(files)):
            # Load the image
            image_path = os.path.join(TO_SEGMENT_DIR, files[file_num])
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                print("Segmentation process found empty file, skipping")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Segment the image
            masks = mask_generator.generate(img)
            # Save the segmentation results
            results.append(
                masks
            )
        return masks
    except Exception as e:
        print("Error in segment_images", e)

def ocr_images(files: list[str]):
    """
    Performs OCR on images using PaddleOCR and returns the OCR results.

    :param files: List of image file names/paths to OCR. Images are assumed to be
    JPEG in this function. Files are assumed to exist in the TO_OCR_DIR. Each file
    should have a JSON file in the SEGMENT_LOCAL_CACHE directory with the same name
    containing the segmentation results for the individual image.
    """
    try:
        print("Loading OCR model")
        ocr = PaddleOCR(
            lang="en",
            show_log=False,
            use_gpu=True,
            use_angle_cls=True,
            det_db_unclip_ratio=2.0,
            det_db_box_thresh=0.5,
            det_limit_side_len=1600,
            det_db_score_mode="slow",
            # Detection
            max_batch_size=OCR_BATCH_SIZE,
            # Recognition
            rec_batch_num=OCR_REC_SIZE,
            max_text_length=64,
        )
        final_results = []
        for file_num in range(len(files)):
            # Load the cached segmentation results
            with open(
                os.path.join(
                    TO_OCR_DIR, files[file_num].replace(".jpg", ".json")
                ),
                "r",
            ) as f:
                segmentation_results = json.load(f)
            # Load the image
            image_path = os.path.join(TO_OCR_DIR, files[file_num])
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ocr_results = {}
            # Run the OCR on each segment
            for seg_i, annotation in enumerate(segmentation_results):
                masked_img = apply_mask_output(original_img.copy(), annotation)
                if masked_img is not None:
                    # include 4 rotations
                    sub_results = {}
                    images = [
                        masked_img,
                        cv2.rotate(masked_img, cv2.ROTATE_90_CLOCKWISE),
                        cv2.rotate(masked_img, cv2.ROTATE_180),
                        cv2.rotate(masked_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    ]
                    directions = ["0", "90", "180", "270"]
                    for i, img in enumerate(images):
                        result = ocr.ocr(img, cls=True)
                        sub_results[directions[i]] = result
                    ocr_results[seg_i] = sub_results
            final_results.append(
                ocr_results
            )
        return final_results
    except Exception as e:
        print("Error in ocr_images", e)
