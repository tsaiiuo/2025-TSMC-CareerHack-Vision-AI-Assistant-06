import numpy as np
import utils
import base64
from PIL import Image
import os

def is_base64(s):
    try:
        # Decode the string and then encode it back to compare
        # This ensures the string is a valid Base64 and it didn't have extra characters
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def check_for_fail(metric_param_1, metric_param_2):
    if isinstance(metric_param_1, str) and (metric_param_1.lower() == "fail" or metric_param_1.lower() == "timeout"):
        return True
    elif isinstance(metric_param_2, str) and (metric_param_2.lower() == "fail" or metric_param_2.lower() == "timeout"):
        return True
    else:
        return False

def calculate_iou(ground_truth_mask_path, submitted_answer_path, ground_truth_root, submitted_answer_root):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    :param ground_truth_mask_path: Path to the first binary mask image.
    :param submitted_answer_path: Path to the second binary mask image.
    :return: IoU value.
    """
    if check_for_fail(ground_truth_mask_path, submitted_answer_path):
        return 0
    else:
        ground_truth_mask_path= os.path.join(ground_truth_root, ground_truth_mask_path)
        # Load the mask images
        ground_truth_mask = Image.open(ground_truth_mask_path).convert('L')
        if is_base64(submitted_answer_path):
            submitted_answer_path = utils.decode_base64(submitted_answer_path)
        else:
            submitted_answer_path= os.path.join(submitted_answer_root, submitted_answer_path)
        submitted_answer = Image.open(submitted_answer_path).convert('L')

        # Ensure the masks have the same size
        assert ground_truth_mask.size == submitted_answer.size, "Masks must have the same size"

        # Convert the images to numpy arrays
        ground_truth_mask_array = np.array(ground_truth_mask) // 255  # Convert to binary (0 or 1)
        submitted_answer_array = np.array(submitted_answer) // 255  # Convert to binary (0 or 1)

        # Calculate intersection and union
        intersection = np.logical_and(ground_truth_mask_array, submitted_answer_array).sum()
        union = np.logical_or(ground_truth_mask_array, submitted_answer_array).sum()

        # Calculate IoU
        iou = intersection / union if union != 0 else 0

        return iou

def calculate_MAE(ground_truth, submitted_answer):
    if check_for_fail(ground_truth, submitted_answer):
        return 1
    else:
        return np.abs(float(ground_truth) - float(submitted_answer)) / float(ground_truth)