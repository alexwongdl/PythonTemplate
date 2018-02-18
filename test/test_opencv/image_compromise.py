"""
Created by Alex Wang
On 2018-01-30
"""
import cv2
import numpy as np

def test_arithmetic_opt():
    """
    测试图像融合
    :return:
    """
    google_logo = cv2.imread('google.jpg')
    dl_img = cv2.imread('dl.jpg')

    # addWeighted
    google_resize = np.ones(dl_img.shape, np.uint8)
    google_resize = google_resize * 255
    google_resize[0:google_logo.shape[0], 0:google_logo.shape[1]] = google_logo

    simple_add_img = cv2.addWeighted(dl_img, 0.7, google_resize, 0.5, 0)  # 需要相同的大小
    cv2.imshow('google_resize', google_resize)
    cv2.imshow('add_weighted', simple_add_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # add
    google_logo_gray = cv2.cvtColor(google_resize, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(google_logo_gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    google_logo_fg = cv2.bitwise_and(google_resize, google_resize, mask = mask_inv)
    dl_bg = cv2.bitwise_and(dl_img, dl_img, mask = mask)

    added_img = cv2.add(google_logo_fg, dl_bg)
    cv2.imshow('mask', mask)
    cv2.imshow('mask_inv', mask_inv)
    cv2.moveWindow('mask_inv',200,200)
    cv2.imshow('google_logo_fg', google_logo_fg)
    cv2.imshow('dl_bg', dl_bg)
    cv2.imshow('added_img',added_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_arithmetic_opt()