import tensorflow as tf
import numpy as np
from inception import image_processing
from inception import inception_model as inception
from scipy import misc
import numpy
import PIL

def eval_image(image, height, width, scope=None):
    """
    图像预处理
    Prepare one image for evaluation.
    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(values=[image, height, width], name=scope,
                       default_name='eval_image'):
        ## 像素值转换到 [0,1] 范围 tf.uint8到tf.float32的转换
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])

        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

####################################################  加载已经训练好的模型  ################################################
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state('/home/recsys/hzwangjian1/tensorflow/models/inception/darthvader_model')
ckpt = tf.train.get_checkpoint_state('/home/recsys/hzwangjian1/tensorflow/models/inception/inception/flower_model')
print(ckpt.model_checkpoint_path)

images_input = tf.placeholder(tf.float32,shape = (1,299,299,3))
logits, _ = inception.inference(images_input, 6)
# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, ckpt.model_checkpoint_path)


####################################################  载入一张图片来测试模型  ################################################
img = misc.imread( "/home/recsys/hzwangjian1/tensorflow/models/inception/inception/data/raw-data/validation/dandelion/2465442759_d4532a57a3.jpg")
import matplotlib.pyplot as plt
fig = plt .figure()
plt.subplot(1,2,1)
plt.imshow (img)

print(img.shape)
img_tensor = tf.convert_to_tensor(img)
img_tensor_corp = eval_image(img_tensor, 299, 299)
img_tensor_val = sess.run(img_tensor_corp)
print(img_tensor_val.shape)

img_list = []
img_list.append(img_tensor_val)

logitss_val = sess.run(logits, feed_dict={images_input: img_list})
print(logitss_val)

plt.subplot(1,2,2)
plt.imshow (img_tensor_val)
plt.show ()
