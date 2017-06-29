
# coding: utf-8

# ## 图片重复检测神经网络

# In[29]:

import concurrent.futures
from scipy import misc
import numpy as np
import pickle
from PIL import Image
import requests
from io import StringIO
from io import BytesIO
import threading
import _thread
import codecs
import json
import multiprocessing
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Queue
import matplotlib.pyplot as plt
import tensorflow as tf

import sklearn
from sklearn import cross_validation
import random 


# ## 全局变量

# In[31]:

tfrecords_file = "/home/recsys/hzwangjian1/data/train.tfrecords"
file_to_write = "/home/recsys/hzwangjian1/data/imagepath_pair_duplabel.data"


# In[32]:


# 获取《图片一路径，图片二路径，标记》数据对
def load_data():
    file_to_write = "/home/recsys/hzwangjian1/data/imagepath_pair_duplabel.data"
    reader_handler = open(file_to_write, 'r')

    image_one_path_list = []
    image_two_path_list = []
    label_list = []

    count = 0
    for line in reader_handler:
        count = count + 1
        elems = line.split("\t")
        if len(elems) < 3:
            print("len(elems) < 3:" + line)
            continue
        image_one_path = elems[0].strip()
        image_two_path = elems[1].strip()
        label = int(elems[2].strip())
#         if label == 0:
#             label = -1

        image_one_path_list.append(image_one_path)
        image_two_path_list.append(image_two_path)
        label_list.append(label)


    print(len(image_one_path_list))
    print(len(image_two_path_list))
    print(len(label_list))
    return image_one_path_list, image_two_path_list, label_list



# ## map多线程、多进程

# In[33]:

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
def creat_thumbnail(image_path):
    img = misc.imread(image_path)
    img_arr = np.asarray(img)
    return img_arr

def load_image_with_path_list(image_path_list):
#     pool = Pool()
    pool = ThreadPool(30)
    image_list = pool.map(creat_thumbnail, image_path_list)
    pool.close()
    pool.join()
    
    return image_list


# In[34]:

tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 50
display_step = 10

# Network Parameters
image_width = 256
image_height = 256
image_channel = 3
n_input = image_width * image_height

# tf Graph input
with tf.name_scope('input_data') as scope:
    X1 = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name='image_one')
    X2 = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name='image_two')
    y = tf.placeholder(tf.float32, [None], name='label')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='drop_out') #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

variables_dict = {
    "W_conv1":tf.Variable(tf.random_normal(shape=[11,11, image_channel,32]), name='weight'),
    "b_conv1":tf.Variable(tf.random_normal(shape=[1,32]), name='bias'),
    
    "W_conv2":tf.Variable(tf.random_normal(shape=[5,5,32,64]), name='weight'),
    "b_conv2":tf.Variable(tf.random_normal( shape=[1,64]), name='bias'),
    
    "W_conv3":tf.Variable(tf.random_normal(shape=[3,3,64,64]), name='weight'),
    "b_conv3":tf.Variable(tf.random_normal(shape=[1,64], name='bias')),
    
    "W_full":tf.Variable(tf.random_normal(shape=[8 * 8 * 64, 1024]), name='weight'),
    "b_full":tf.Variable(tf.random_normal(shape=[1, 1024]), 'bias')
}

# Create model
def conv_net(x,dropout):
    with tf.name_scope('model') as scope:
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, image_width, image_height, image_channel])

        # Convolution Layer

        # Max Pooling (down-sampling)
        with tf.name_scope('layer1') as scope:
            W_conv1 = variables_dict["W_conv1"]
            b_conv1 = variables_dict["b_conv1"]
            convOne = tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1], padding="SAME")
            reluOne = tf.nn.relu(convOne + b_conv1)
            conv1 = tf.nn.max_pool(reluOne, ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

        # Convolution Layer
        with tf.name_scope('layer2') as scope:
            W_conv2 = variables_dict["W_conv2"]
            b_conv2 = variables_dict["b_conv2"]
            convTwo = tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding="SAME")
            reluTwo = tf.nn.relu(convTwo + b_conv2)
            conv2 = tf.nn.max_pool(reluTwo, ksize=[1,4,4,1], strides=[1,4,4,1],padding="SAME")

        with tf.name_scope('layer3') as scope:
            W_conv3 = variables_dict["W_conv3"]
            b_conv3 = variables_dict["b_conv3"]
            convThree = tf.nn.conv2d(conv2, W_conv3, strides=[1,1,1,1], padding="SAME")
            reluThree = tf.nn.relu(convThree + b_conv3)
            conv3 = tf.nn.max_pool(reluThree, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
            
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        with tf.name_scope('full_connect') as scope:
            W_full = variables_dict["W_full"]
            b_full = variables_dict["b_full"]
            input_flat=tf.reshape(conv3, shape=[-1, 8 * 8 * 64])
            fc1 = tf.nn.relu(tf.matmul(input_flat, W_full) + b_full)

        # Apply Dropout
        with tf.name_scope('dropout_layer') as scope:
            drop_out = tf.nn.dropout(fc1,keep_prob)

        norm_fc1 = tf.reduce_sum(tf.mul(fc1,fc1),reduction_indices=1)
    
        return fc1

with tf.name_scope('whole_model') as scope:
    img_one_rep = conv_net(X1, dropout=keep_prob)
    img_two_rep = conv_net(X2, dropout=keep_prob)
    
    norm_one_rep = tf.sqrt(tf.reduce_sum(tf.mul(img_one_rep,img_one_rep),reduction_indices=1))
    norm_two_rep = tf.sqrt(tf.reduce_sum(tf.mul(img_two_rep, img_two_rep),reduction_indices=1))
    norm_mul = tf.mul(norm_one_rep, norm_two_rep)
    
    cos_rep = tf.div(tf.reduce_sum(tf.mul(img_one_rep, img_two_rep),reduction_indices=1), norm_mul)
    
with tf.name_scope('result') as scope:
    with tf.name_scope('norm_W'):
        norm_W1 = tf.sqrt(tf.reduce_sum(tf.mul(variables_dict["W_conv1"], variables_dict["W_conv1"])))
        norm_W2 = tf.sqrt(tf.reduce_sum(tf.mul(variables_dict["W_conv2"], variables_dict["W_conv2"])))
        norm_W3 = tf.sqrt(tf.reduce_sum(tf.mul(variables_dict["W_conv3"], variables_dict["W_conv3"])))
        norm_full = tf.sqrt(tf.reduce_sum(tf.mul(variables_dict["W_full"], variables_dict["W_full"])))
        norm_W = norm_W1 + norm_W2 + norm_W3 + norm_full
        
#     cross_entropy_cnn = -(y * tf.nn.log_softmax(cos_rep) + (1-y) * tf.nn.log_softmax(1 - cos_rep))
    cross_entropy_cnn = tf.nn.sigmoid_cross_entropy_with_logits(logits=cos_rep, targets=y)
    cost =tf.reduce_sum(cross_entropy_cnn, name='cost') + 0.01 * norm_W
        
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    cos_rep_gt_zero = tf.greater(cos_rep, 0)
    label_gt_zero = tf.greater(y, 0)
    correct_pred = tf.equal(cos_rep_gt_zero, label_gt_zero)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# ## 主入口

# In[35]:

# 获取《图片一，图片二，标记》数据对 ， label等于 0 或 +1
image_one_path_list, image_two_path_list, label_list = load_data()  


# 数据划分
img_one_train, img_one_test, img_two_train, img_two_test, label_train, label_test = cross_validation.train_test_split(image_one_path_list, image_two_path_list, label_list, test_size= 0.2)
print(len(img_one_test))
print(len(img_two_test))
print(len(label_test))

# 获取图片数据
# print("获取图片数据")
# images_one = load_image_with_path_list(img_one_train)
# print(len(images_one))

# images_two = load_image_with_path_list(img_two_train)
# print(len(images_two))

# images_test_one = load_image_with_path_list(img_one_test)
# images_test_two = load_image_with_path_list(img_two_test)



# In[36]:

def get_image(image_path):  
    """Reads the jpg image from image_path. 
    Returns the image as a tf.float32 tensor 
    Args: 
        image_path: tf.string tensor 
    Reuturn: 
        the decoded jpeg image casted to float32 
    """  
    content =tf.read_file(image_path)
    return tf.image.decode_jpeg(content, channels=3)

## RunnerQueue
train_input_queue = tf.train.slice_input_producer( [img_one_train, img_two_train, label_train], shuffle=True, capacity=4 * batch_size)  
img_one_queue = get_image(train_input_queue[0])
img_two_queue = get_image(train_input_queue[1])
label_queue = train_input_queue[2]

batch_img_one, batch_img_two, batch_label = tf.train.shuffle_batch([img_one_queue, img_two_queue, label_queue],                                                                   batch_size=batch_size,capacity =  10 + 3* batch_size,                                                                   min_after_dequeue = 10,num_threads=16,                                                                  shapes=[(image_width, image_height, image_channel),                                                                          (image_width, image_height, image_channel),()])


# In[37]:

cost_summary = tf.scalar_summary("cost", cost)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
norm_summary = tf.scalar_summary("norm_W", norm_W)
# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
# with tf.Session() as sess:
sess.run(tf.initialize_all_variables())


# img_one_sample_tf = tf.reshape(tf.cast(images_one[8], tf.uint8), shape=[-1, image_width, image_height, image_channel])
# image_summary_op = tf.image_summary("img_one_sample", img_one_sample_tf)
image_summary_one_op = tf.image_summary("img_one_sample", tf.reshape(batch_img_one, shape=[-1, image_width, image_height, image_channel]))

summary_op = tf.merge_summary([cost_summary, accuracy_summary, norm_summary])
summary_writer = tf.train.SummaryWriter('/home/recsys/hzwangjian1/tensorboard/test', graph_def=sess.graph)


coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess,coord=coord)  

for i in range(100):

    batch_img_one_val, batch_img_two_val, label = sess.run([batch_img_one, batch_img_two,batch_label])
    print(batch_img_one_val.shape)
    print(label)
    
    if i% 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={X1:batch_img_one_val, X2:batch_img_two_val, y:label, keep_prob:1.0},session=sess)
            print ("step "+ str(i) +", training accuracy :"+ str(train_accuracy))
            cross_entropy_val = cross_entropy_cnn.eval({X1:batch_img_one_val, X2:batch_img_two_val, y:label, keep_prob:1.0},session=sess)

            summary_str = sess.run(summary_op, feed_dict={X1:batch_img_one_val, X2:batch_img_two_val, y:label, keep_prob:(1.0)})
            summary_writer.add_summary(summary_str, i)

    sess.run([optimizer,summary_op], feed_dict={X1:batch_img_one_val, X2:batch_img_two_val, y:label, keep_prob:0.75})

#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(batch_img_one_val[1])
#     fig.add_subplot(1,2,2)
#     plt.imshow(batch_img_two_val[1])
#     plt.show()
    

# print("test accuracy :" + str(accuracy.eval(feed_dict={X1:images_test_one[1:500], X2:images_test_two[1:500] ,y:label_test[1:500], keep_prob:1.0},session=sess)))


coord.request_stop()  
coord.join(threads)  

sess.close()
summary_writer.close()
print("all done")


# In[ ]:


# del images_one
# del images_two

index = 98
print(images_one[index].shape)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(images_one[index])
fig.add_subplot(1,2,2)
plt.imshow(images_two[index])
plt.show()

print(len(images_one[1:5]))

print(label_train[index])

index_test = 98
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(images_test_one[index_test])
fig.add_subplot(1,2,2)
plt.imshow(images_test_two[index_test])
plt.show()

print(label_test[index_test])


print( random.randint(0,len(img_one_train) - batch_size))

