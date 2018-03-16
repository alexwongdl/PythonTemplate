"""
Created by Alex Wang on 2018-03-15
"""
import cv2
import os

print(os.getcwd())  ##当前目录


def opencv_face_detect():
    """
    opencv haar cascade face detection
    :return:
    """
    i = 0
    img_path = os.path.join(os.getcwd(), 'data', 'running_man.jpg')
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    print(dir(face_cascade))
    print(help(face_cascade.detectMultiScale))
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 3)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
        roi_gray = img_gray[y: y + h, x:x + w]
        roi_img = img[ y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), color=(0, 0, 255), thickness=3)
        # cv2.imshow('img_' + str(i), roi_img.copy())
        # i += 1

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opencv_face_detect()
