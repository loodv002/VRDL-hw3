import os
import tifffile
import numpy as np
import cv2
from pycocotools import mask as mask_utils

def read_mask(mask_path) -> np.ndarray:
    return tifffile.imread(mask_path).astype(np.uint16)

def show_mask(mask_path):
    assert os.path.exists(mask_path)

    image = tifffile.imread(mask_path)
    h, w = image.shape
    image = cv2.resize(image, (w//2, h//2))
    cv2.imshow('', image)
    cv2.waitKey()

def encode_mask(binary_mask: np.ndarray):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def count_mask_instances(mask_path) -> int:
    assert os.path.exists(mask_path)

    image = tifffile.imread(mask_path).astype(np.uint16)
    return np.unique(image).size - 1
