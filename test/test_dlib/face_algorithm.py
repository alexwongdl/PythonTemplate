"""
Created by Alex Wang
On 2018-05-03

CNN Face Detector
Face Alignment
Face Clustering
Face Detector
Face Jittering/Augmentation
Face Landmark Detection
Face Recognition
"""

import dlib
from skimage import io
import cv2


def test_cnn_face_detector():
    """
    You can get the mmod_human_face_detector.dat file from:\n
    http://dlib.net/files/mmod_human_face_detector.dat.bz2
    :return:
    """
    img_path = 'data/running_man.jpg'
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    img = io.imread(img_path)
    # upsample image 1 time, make everything bigger and allow us to detect more faces.
    dets = cnn_face_detector(img, 2)

    img_cv2 = cv2.imread(img_path)
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        cv2.rectangle(img_cv2, (d.rect.left(), d.rect.top()),
                      (d.rect.right(), d.rect.bottom()), color=(255, 0, 0), thickness=3)
    cv2.imshow('img', img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_hog_face_detector():
    """
    blog:
        https://blog.gtwang.org/programming/python-opencv-dlib-face-detection-implementation-tutorial/
    detector idx:
        https://blog.gtwang.org/programming/python-opencv-dlib-face-detection-implementation-tutorial/
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hog_face_detector = dlib.get_frontal_face_detector()
    # 第二个参数--upsample, 第三个参数--threshold,超过threshold的结果会被输出
    # returns:rects--检测框位置,scores--得分,idx--检测器编号(正脸/侧脸等)
    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)

    for i, rect in enumerate(rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        cv2.putText(img, '{:.2f}({})'.format(scores[i], idx[i]), (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_alignment():
    """
    input: image including faces
    output:aligned faces

    You can get the shape_predictor_5_face_landmarks.dat from:
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hog_face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(shape_predictor(img_rgb, rect))

    # get the aligned face images
    images = dlib.get_face_chips(img, faces, size=320)
    for ind, image in enumerate(images):
        print(ind)
        # image_patch = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('patch{}'.format(ind), image)

    # get a single chip
    print('faces[0]', faces[0])
    image = dlib.get_face_chip(img, faces[0], size=80)
    cv2.imshow('single chip', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_alignment_cnn():
    """
    input: image including faces
    output:aligned faces

    You can get the shape_predictor_5_face_landmarks.dat from:
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    height, width = img.shape[0:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    dets = cnn_face_detector(img, 2)
    faces = dlib.full_object_detections()
    print(help(faces))
    print(dir(faces))
    for i, d in enumerate(dets):
        # rect = d.rect
        # x, y = rect.left(), rect.top()
        # w = rect.right() - x
        # h = rect.bottom() - y
        # up_margin = w / 4
        # margin = w / 5
        # y_min = int(max(0, y - up_margin))
        # y_max = int(min(height, y + h + margin))
        # x_min = int(max(0, x - margin))
        # x_max = int(min(width, x + w + margin))
        # rect_new = dlib.rectangle(left=x_min, top=y_min, right=x_max, bottom=y_max)
        faces.append(shape_predictor(img_rgb, d.rect))

    # get the aligned face images
    images = dlib.get_face_chips(img, faces, size=80)
    for ind, image in enumerate(images):
        print(ind)
        # image_patch = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('patch{}'.format(ind), image)

    # get a single chip
    print('faces[0]', faces[0])
    image = dlib.get_face_chip(img, faces[0], size=80)
    cv2.imshow('single chip', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_jitter():
    """
    This example shows how faces were jittered and augmented to create training
    data for dlib's face recognition model.  It takes an input image and
    disturbs the colors as well as applies random translations, rotations, and
    scaling.
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get face
    hog_face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(shape_predictor(img_rgb, rect))

    face_image = dlib.get_face_chip(img_rgb, faces[0], size=80)

    # jitter face
    jittered_images = dlib.jitter_image(face_image, num_jitters=4, disturb_colors=True)
    for idx, image in enumerate(jittered_images):
        image_brg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('jittered_image_{}'.format(idx), image_brg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_landmarks():
    """
    Estimate the pose with 68 landmarks.The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
        https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
        C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
        300 faces In-the-wild challenge: Database and results.
        Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016
    You can download a trained facial shape predictor from :
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    :return:
    """
    img_path = 'data/running_man.jpg'
    img_path = 'data/face_one.png'
    # img_path = 'data/face_two.jpg'
    img = cv2.imread(img_path)
    print('img_shape:', img.shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get face
    hog_face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(shape_predictor(img_rgb, rect))

    for landmark in faces:
        for idx, point in enumerate(landmark.parts()):
            # cv2.putText(img, '*', (point.x, point.y), cv2.FONT_HERSHEY_DUPLEX, 0.1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(idx), (point.x, point.y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('face_img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_recognition():
    """
    Map an image of a human face to a 128d vector space.
    You can perform face recognition by checking if their Euclidean distance is small enough.

    When using a distance threshold of 0.6, the dlib model obtains an accuracy
    of 99.38% on the standard LFW face recognition benchmark

    You can download a trained facial shape predictor and recognition model from:
        http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
        http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

    algorithm:
        http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
        [1] O. M. Parkhi, A. Vedaldi, A. Zisserman Deep Face Recognition British Machine Vision Conference, 2015.
        [2] H.-W. Ng, S. Winkler. A data-driven approach to cleaning large face datasets. Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get face
    hog_face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    face_feature_extractor = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    rects, scores, idx = hog_face_detector.run(img_rgb, 2, 0)
    for rect in rects:
        shape = shape_predictor(img_rgb, rect)
        # dlib.vector
        face_descriptor = face_feature_extractor.compute_face_descriptor(img, shape)
        print('face_descriptor', list(face_descriptor))


if __name__ == '__main__':
    # test_cnn_face_detector()
    # test_hog_face_detector()
    # test_face_alignment()
    test_face_alignment_cnn()
    # test_face_jitter()
    # test_face_landmarks()
    # test_face_recognition()
