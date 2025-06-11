import cv2
import numpy as np
from pycocotools import mask as mask_utils


def apply_mask_color(img, annotation, color: tuple[int, int, int] =(0, 255, 0), alpha: float =0.3):
    """
    Apply a mask to an image with a specific color and alpha value.
    Apply a border around the mask with black color.

    :param img: The input image as a NumPy array such as a cv2 image.
    :param annotation: The annotation containing the segmentation mask and bounding box
    returned by COCO API or SAM model
    :param color: The color to apply to the mask (default is green).
    :param alpha: The alpha value for the mask overlay (default is 0.3).
    """
    mask = mask_utils.decode(annotation["segmentation"])
    mask = mask.astype(np.uint8)
    img_masked = img.copy()
    # Apply mask
    img_masked[mask == 1] = (
        img_masked[mask == 1] * (1 - alpha) + np.array(color) * alpha * 255
    )
    # Apply border
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_masked, contours, -1, (0, 0, 0), 2)
    return img_masked


def apply_mask_output(img, annotation):
    """
    Apply a mask to an image and extract the masked region based on the bounding box.

    :param img: The input image as a NumPy array such as a cv2 image.
    :param annotation: The annotation containing the segmentation mask and bounding box
    returned by COCO API or SAM model
    """
    try:
        # Extract the mask
        mask = mask_utils.decode(annotation["segmentation"])
        mask = mask.astype(np.uint8)
        # Apply to image
        img_masked = np.where(mask[..., None], img, 255)
        x, y, w, h = annotation["bbox"]
        # Convert coords to int
        x, y, w, h = int(x), int(y), int(w), int(h)
        img_cropped = img_masked[y : y + h, x : x + w]
        # Only return the cropped image if it's not empty
        if img_cropped.size == 0:
            return None
        return img_cropped
    except Exception as e:
        print("Error in apply_mask_output", e)
        print("Shapes (img, mask):", img.shape, mask.shape)
        print(annotation)
        return None
