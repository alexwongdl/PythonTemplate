"""
Created by Alex Wang
On 2018-01-30
"""
import sys

import cv2
import numpy as np


def test_imshow():
    """
    测试opencv图像展示
    :return:
    """
    img = cv2.imread('E://blog/oxford_pet.png')
    cv2.imshow('window1', img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_waitKey():
    """
    使用waitKey函数进行用户交互
    :return:
    """
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        sys.exit(0)
    elif key == ord('q'):
        return


def test_rectangle_putText():
    """
    Test cv2.rectangle and cv2.putText
    :return:
    """
    colours = np.random.rand(32, 3)  # used only for display
    image_org = cv2.imread('data/dl.jpg')
    d = [0, 1, 80, 90, 1]
    cv2.rectangle(image_org, (d[0], d[1]), (d[2], d[3]),
                  color=colours[d[4] % 32, :],
                  thickness=2)
    cv2.putText(image_org, str(d[4]), (d[0], d[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                color=(0, 0, 255),
                thickness=2
                )


def rectangle(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)

def rectangle_bottom(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        # center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        # pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        # right-bottom cornor
        center = pt2[0] - 5 - text_size[0][0], pt2[1] + 5 - text_size[0][1]
        pt1 = pt2[0] - 10 - text_size[0][0], pt2[1] - 10 - text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)

if __name__ == '__main__':
    test_imshow()
