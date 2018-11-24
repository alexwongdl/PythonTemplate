"""
Created by Alex Wang
on 2018-09-26
"""
import base64

import numpy as np
import tensorflow as tf
import cv2


def test_base64():
    # numpy --> base64 --> numpy
    t = np.arange(25, dtype=np.float32)
    t = np.reshape(t, newshape=(5, 5))
    s = base64.urlsafe_b64encode(t)
    r = base64.decodestring(s)
    q = np.frombuffer(r, dtype=np.float32)
    print(t)
    print(q)

    # img = cv2.imread('data/laska.png')
    # img_str = base64.urlsafe_b64encode(img)
    # img_decode = base64.decodestring(img_str)
    # img_np = np.frombuffer(img_decode, dtype=np.uint8)
    # img_q = cv2.imdecode(img_np, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img_q', img_q)

    # numpy --> image_str --> base64 --> image_str --> numpy
    img = cv2.imread('data/laska.png')
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    img_encode = base64.urlsafe_b64encode(img_str)

    frame = tf.placeholder(tf.string)
    image = tf.decode_base64(frame)
    rgb = tf.image.decode_jpeg(image, channels=3)

    with tf.Session() as sess:
        rgb_image = sess.run(rgb, feed_dict={frame: img_encode})

    cv2.imshow('img', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(np.allclose(q, t))


import PIL
from PIL import Image
from io import BytesIO


# import cStringIO for python 2


def test_base64_pillow():
    img = cv2.imread('data/laska.png')
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    buffer = BytesIO()  # buffer = cStringIO.StringIO() for python2
    pil_img.save(buffer, format="JPEG", quality=100)
    b64code = base64.b64encode(buffer.getvalue())

    img_base64_data = base64.b64decode(b64code)
    img_nparr = np.fromstring(img_base64_data, np.uint8)
    img = cv2.imdecode(img_nparr, cv2.COLOR_BGR2RGB)

    print(img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def opencv_img_to_string():
    """
    使用imencode、imdecode
    :return:
    """
    img = cv2.imread('data/laska.png')
    img_str = cv2.imencode('.jpg', img)[1].tostring()  # 将图片格式转换(编码)成流数据，放到内存缓存中，然后转化成string格式
    b64_code = base64.b64encode(img_str)

    str_decode = base64.b64decode(b64_code)
    nparr = np.fromstring(str_decode, np.uint8)
    # img_restore = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) for python 2
    img_restore = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imshow('img', img_restore)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def numpy_arr_to_string():
    """
    使用BytesIO，仅限于1维或者2维矩阵
    :return:
    """
    arr = np.arange(12).reshape(3, 4)
    bytesio = BytesIO()
    np.savetxt(bytesio, arr)
    content = bytesio.getvalue()
    print(content)

    b64_code = base64.b64encode(content) b64_code = base64.urlsafe_b64encode(content)
    b64_decode = base64.b64decode(b64_code) # b64_decode = base64.urlsafe_b64decode(b64_code)

    arr = np.loadtxt(BytesIO(b64_decode))
    print(arr)


if __name__ == '__main__':
    # test_base64()
    # test_base64_pillow()
    # opencv_img_to_string()
    numpy_arr_to_string()