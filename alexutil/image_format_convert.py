"""
Created by Alex Wang on 2019-03-04
"""

import cv2
from PIL import Image
import numpy as np

def pil_to_cvmat(pil_image):
    """
    convert PIL image to opencv image 
    :param pil_image:
    :return:
    """
    opencv_img = np.array(pil_image)
    # Convert RGB to BGR
    opencv_img = opencv_img[:, :, ::-1]
    return opencv_img


