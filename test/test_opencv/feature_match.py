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
    scenery_part = scenery_img[500:720, 700:1024, :]
    scenery_part_resize = cv2.resize(scenery_part, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
    print(scenery_part_resize.shape)
    scenery_part_rows, scenery_part_cols, _ = scenery_part_resize.shape

    rows_bg, cols_bg, channels_bg = scenery_bg.shape
    black_img = np.zeros(shape=(rows_bg, cols_bg, channels_bg), dtype=np.uint8)
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
    # query_img = cv2.imread('scenery_combined.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    orb = cv2.ORB_create()

    kp_query, des_query = orb.detectAndCompute(query_img, None)
    kp_base, des_base = orb.detectAndCompute(base_img, None)

    help(cv2.BFMatcher)
    # help(cv2.BFMatcher_create())

    # method one
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = bfMatcher.match(des_query, des_base)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, matches[0:50], None, flags=2)

    # method similar to sift does not work
    # good_matches =[]
    # for m,n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    # img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, good_matches, None, flags=2)

    # method two , support multiple images match
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # crossCheck should be false
    bfMatcher.add([des_base])
    bfMatcher.add([des_base])
    # bfMatcher.train() # empty implementation for BruteForceMatcher
    matches = bfMatcher.knnMatch(des_query, 3)  # also support match and radiusMatch

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


def test_BFMatcher_SIFT():
    """
    Brute Force Matcher for SIFT
    :return:
    """
    query_img = cv2.imread('scenery.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    query_kp, query_des = sift.detectAndCompute(query_img_gray, None)
    base_kp, base_des = sift.detectAndCompute(base_img_gray, None)

    bfMatcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = bfMatcher.knnMatch(query_des, base_des, k=2)

    good_matches = []
    for m, n in matches:  # m:最近的特征，n：第二近的特征
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(query_img, query_kp, base_img, base_kp, good_matches, None, flags=2)
    cv2.imshow('img_matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_KDTree_SIFT():
    """
    FLANN KD树检索SIFT特征
    :return:
    """
    query_img = cv2.imread('scenery.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(query_img_gray, None)
    kp_base, des_base = sift.detectAndCompute(base_img_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)  # how many leafs to check in one search
    kdtree = cv2.FlannBasedMatcher(index_params, search_params)
    matches = kdtree.knnMatch(des_query, des_base, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, good_matches, None, flags=2)
    cv2.imshow('img_matcher', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_LSH_ORB():
    """
    FLANN LSH检索ORB特征
    :return:
    """
    FLANN_INDEX_LSH = 6
    query_img = cv2.imread('scenery.jpg')
    # query_img = cv2.imread('scenery_combined.jpg') # good performance
    base_img = cv2.imread('scenery_combined.jpg')
    orb = cv2.ORB_create()

    kp_query, des_query = orb.detectAndCompute(query_img, None)
    kp_base, des_base = orb.detectAndCompute(base_img, None)

    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=16, multi_probe_level=2)
    search_params = dict()
    lsh = cv2.FlannBasedMatcher(index_params, search_params)
    print(dir(lsh))
    # lsh.match(des_query, des_base)

    lsh.add([des_base])
    lsh.add([des_base])
    lsh.train()

    matches_one = []
    matches = lsh.knnMatch(des_query, k=2)
    for m,n in matches:
        if m.imgIdx == 0:
            matches_one.append(m)
        if n.imgIdx == 0:
            matches_one.append(n)

    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, matches_one, None, flags=2)
    cv2.imshow('img_matcher', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(len(matches))
def test_KDTree_SURF():
    """
    FLANN KD树检索SURF特征   m.distance < 0.75 * n.distance
    :return:
    """
    query_img = cv2.imread('scenery.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(400)
    kp_query, des_query = surf.detectAndCompute(query_img_gray, None)
    kp_base, des_base = surf.detectAndCompute(base_img_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)  # how many leafs to check in one search
    kdtree = cv2.FlannBasedMatcher(index_params, search_params)
    matches = kdtree.knnMatch(des_query, des_base, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # homography
    query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    base_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    print('query_pts shape:', query_pts.shape) # (183, 2)
    M, mask = cv2.findHomography(query_pts, base_pts, cv2.RANSAC, 5.0)
    mask_list = mask.ravel()
    filted_matches = []
    for i in range(len(good_matches)):
        if mask_list[i] == 1:
            filted_matches.append(good_matches[i])
    print('filted_matches number:', len(filted_matches))

    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, filted_matches, None, flags=2)
    cv2.imshow('img_matcher', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_KDTree_SURF_1():
    """
    FLANN KD树检索SURF特征   m.distance <= max(3 * min_dist, 0.02):
    :return:
    """
    query_img = cv2.imread('scenery.jpg')
    base_img = cv2.imread('scenery_combined.jpg')
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(400)
    kp_query, des_query = surf.detectAndCompute(query_img_gray, None)
    kp_base, des_base = surf.detectAndCompute(base_img_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)  # how many leafs to check in one search
    kdtree = cv2.FlannBasedMatcher(index_params, search_params)
    matches = kdtree.match(des_query, des_base)

    good_matches = []
    min_dist = 100
    for m in matches:
        if m.distance < min_dist:
            min_dist = m.distance

    for m in matches:
        if m.distance <= max(3 * min_dist, 0.02):
            good_matches.append(m)

    # homography
    query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    base_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    print('query_pts shape:', query_pts.shape) # (183, 2)
    M, mask = cv2.findHomography(query_pts, base_pts, cv2.RANSAC, 5.0)
    mask_list = mask.ravel()
    filted_matches = []
    for i in range(len(good_matches)):
        if mask_list[i] == 1:
            filted_matches.append(good_matches[i])

    img_matches = cv2.drawMatches(query_img, kp_query, base_img, kp_base, filted_matches, None, flags=2)
    cv2.imshow('img_matcher', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # gen_test_image()
    # test_BFMatcher_ORB()
    # test_BFMatcher_SIFT()
    # test_KDTree_SIFT()
    # test_LSH_ORB()
    test_KDTree_SURF()
    # test_KDTree_SURF_1()