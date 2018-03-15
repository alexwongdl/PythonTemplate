"""
Created by Alex Wang on 2018-03-15
"""
import cv2
import os
print(os.getcwd()) ##当前目录

def opencv_face_detect():
    """
    opencv haar cascade face detection
    :return:
    """
    img_path = os.path.join(os.getcwd(), 'data','running_man.jpg')
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x,y), (x + w, y + h), color=(255, 0,0), thickness = 3)

    cv2.imshow('img', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opencv_face_detect()