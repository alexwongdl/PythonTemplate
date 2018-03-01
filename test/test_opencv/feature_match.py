"""
Created by Alex Wang on 2018-2-27
图像特征匹配
"""
import cv2
import numpy as np
import poisson_resconstruct


def gen_test_image():
    """
    生成训练图片，scenery的一部分复制到scenery1上
    :return:
    """
    scenery_img = cv2.imread('scenery.jpg')
    scenery_bg = cv2.imread('scenery1.jpg')
    print(scenery_img.shape)
    print(scenery_bg.shape)
    scenery_part = scenery_img[500:720, 700:1024,:]
    scenery_part_resize = cv2.resize(scenery_part, None, fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    print(scenery_part_resize.shape)
    scenery_part_rows, scenery_part_cols, _ = scenery_part_resize.shape

    rows_bg, cols_bg, channels_bg = scenery_bg.shape
    black_img = np.zeros(shape =(rows_bg, cols_bg, channels_bg), dtype=np.uint8)
    black_img[200:200 + scenery_part_rows, 500:500 + scenery_part_cols, :] = scenery_part_resize

    rotate_mat = cv2.getRotationMatrix2D((0, 0), 10, 1)
    black_img_rotate = cv2.warpAffine(black_img, rotate_mat, (cols_bg, rows_bg))
    black_img_rotate_gray = cv2.cvtColor(black_img_rotate, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('black_img_rotate.jpg', black_img_rotate)
    # mask = poisson_resconstruct.mask_det('black_img_rotate.jpg')

    _, mask = cv2.threshold(black_img_rotate_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(black_img_rotate, black_img_rotate, mask=mask)
    bg = cv2.bitwise_and(scenery_bg, scenery_bg, mask=mask_inv)

    img_combined = cv2.add(fg, bg)

    cv2.imwrite('scenery_combined.jpg', img_combined)
    cv2.imshow('scenery_part', black_img_rotate)
    cv2.imshow('mask', mask)
    cv2.imshow('img_combined', img_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_BFMatcher_ORB():
    """
    测试Brute-Force Matching
    :return:
    """
    query_img = cv2.imread('scenery.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    orb = cv2.ORB_create()

    kp_query, des_query = orb.detectAndCompute(query_img, None)
    kp_base, des_base = orb.detectAndCompute(base_img, None)

    help(cv2.BFMatcher)
    # help(cv2.BFMatcher_create())

    # method one
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = bfMatcher.match(des_query, des_base)
    matches = sorted(matches, key=lambda x:x.distance)
    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, matches[0:50], None, flags=2)

    # method two , support multiple images match
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # crossCheck should be false
    bfMatcher.add([des_base])
    bfMatcher.add([des_base])
    # bfMatcher.train() # empty implementation for BruteForceMatcher
    matches = bfMatcher.knnMatch(des_query, 3) # also support match and radiusMatch

    imgIdxSet = set()
    for match in matches:
        for m in match:
            imgIdxSet.add(m.imgIdx)
    print('imgIdxSet', imgIdxSet)

    print('matches num:', len(matches))
    print(dir(bfMatcher))
    help(bfMatcher.match)
    help(bfMatcher.knnMatch)
    cv2.imshow('img_matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # gen_test_image()
    test_BFMatcher_ORB()