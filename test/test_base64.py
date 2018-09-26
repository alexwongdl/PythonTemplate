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


if __name__ == '__main__':
    test_base64()