"""
Created by Alex Wang on 2018-03-13
图像复原美化：
    inpaint水印去除
"""
import cv2
import numpy as np


def test_image_inpaint():
    """
    cv2.inpaint(src, inpaintMask, inpaintRadius, flags[, dst]) → dst
    Parameters:

	* src – Input 8-bit 1-channel or 3-channel image.
	* inpaintMask – Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.
	* dst – Output image with the same size and type as src .
	* inpaintRadius – Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
	* flags –
    Inpainting method that could be one of the following:

		* INPAINT_NS Navier-Stokes based method [Navier01]
		* INPAINT_TELEA Method by Alexandru Telea [Telea04].
    :return:
    """
    img = cv2.imread('scenery.jpg')
    print(img.shape)
    img_black = img.copy()
    black_block = np.zeros(shape=(20, 20, 3), dtype=np.uint8)
    img_black[690:710, 100:120, :] = black_block

    white_block = np.ones(shape=(20, 20), dtype=np.uint8)
    mask_image = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    print(mask_image.shape)
    mask_image[690:710, 100:120] = white_block
    img_recovery = cv2.inpaint(img_black, mask_image, 3, cv2.INPAINT_NS)

    cv2.imshow('img', img)
    cv2.imshow('img_black', img_black)
    cv2.imshow('img_recovery', img_recovery)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_image_inpaint()
