import sys
sys.path.append('..')
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import random
import imageio
import urllib.request

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH_TO_CKPT = '/home/recsys/houxiaoxia/wm_pb4/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/recsys/houxiaoxia/wm_data/wm_label_map2.pbtxt'
NUM_CLASSES = 11
IMAGE_FOLDER = 'images'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(graph=detection_graph, config=config)

def down2file(url,filename,folder=IMAGE_FOLDER): 
    if not os.path.exists(IMAGE_FOLDER):
        os.mkdir(IMAGE_FOLDER)
    f=open(folder+'/'+filename,'wb')  
    print('downloading file:') 
    req=urllib.request.Request(url)
    data=urllib.request.urlopen(req).read()  
    f.write(data)  
    f.close()  
    print('download '+filename+' OK!')
    
def wm_remove(img,ymin,xmin,ymax,xmax):
    shape = img.shape
    h = shape[0]
    w = shape[1]
    mask = np.zeros((h,w),dtype=np.uint8)
    if ymin-5<0:
        ymin =0
    else:
        ymin = ymin-5
    if xmin-5<0:
        xmin =0
    else:
        xmin =xmin-5
    if xmax+5>w:
        xmax = w
    else:
        xmax = xmax+5
    if ymax + 12>h:
        ymax = h
    else:
        ymax = ymax + 12
    mask[ymin:ymax,xmin:xmax] = np.ones((ymax-ymin,xmax-xmin),dtype=np.uint8)*255

    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    return dst[ymin:ymax,xmin:xmax,:],[ymin,ymax,xmin,xmax]

def wm_remove2(img,box_map):#it should be removed
    shape = img.shape
    h = shape[0]
    w = shape[1]
    mask = np.zeros((h,w),dtype=np.uint8)
    rec = []
    for box, color in box_map.items():
        ymin, xmin, ymax, xmax = box
    	ymin = int(ymin*h)
    	xmin = int(xmin*w)
    	ymax = int(ymax*h)
    	xmax = int(xmax*w)
        if ymin-5<0:
            ymin =0
        else:
            ymin = ymin-5
        if xmin-5<0:
            xmin =0
        else:
            xmin =xmin-5
        if xmax+5>w:
            xmax = w
        else:
            xmax = xmax+5
        if ymax + 12>h:
            ymax = h
        else:
            ymax = ymax + 12
        mask[ymin:ymax,xmin:xmax] = np.ones((ymax-ymin,xmax-xmin),dtype=np.uint8)*255
        rec.append([ymin,ymax,xmin,xmax])

    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)    
    return dst,rec

def wm_video(video_path): 
    '''''
    detect watermark in one  of the video frames
    remove watermark from all of frames and save to a new video
    '''''
    video_dst = None
    try:
        vid = imageio.get_reader(video_path,'ffmpeg')
        L = vid.get_length()        
        num = int(L/2)
        print("select %d%s frame for watermark detection:"%(num,'th'))
        image = vid.get_data(num)
        h = image.shape[0]
        w = image.shape[1]
        h_tmp = int(h/3)
        image_detec = image[0:h_tmp,:,:]
        image_np_expanded = np.expand_dims(image_detec, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        image_np,class_name,box_map = vis_util.visualize_boxes_and_labels_on_image_array(
                image_detec, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                category_index, use_normalized_coordinates=True, line_thickness=3)
        if len(class_name)>0:
            print('begain remove watermark ',class_name)
            fps = vid.get_meta_data()['fps']
            video_name, ext = os.path.splitext(video_path)
            video_dst = video_name + '-2' + '.mp4' #where to save the new video
            writer = imageio.get_writer(video_dst, fps=fps, macro_block_size=None)
            #for num,im in enumerate(vid):  
            for i in range(0,L,5):
                for box, color in box_map.items():
                    ymin, xmin, ymax, xmax = box
                    ymin = int(ymin*h_tmp)
                    xmin = int(xmin*w)
                    ymax = int(ymax*h_tmp)
                    xmax = int(xmax*w)
                    im = vid.get_data(i)
                    im1 = vid.get_data(i+1)
                    im2 = vid.get_data(i+2)
                    im3 = vid.get_data(i+3)
                    im4 = vid.get_data(i+4)
                    im_tmp = im[0:h_tmp,:,:]
                    wm_rec,rec = wm_remove(im_tmp,ymin,xmin,ymax,xmax)
                    im[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
                    im1[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
                    im2[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
                    im3[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
                    im4[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
                writer.append_data(im)
                writer.append_data(im1)
                writer.append_data(im2)
                writer.append_data(im3)
                writer.append_data(im4)
            writer.close()
        else:
            print('there is no watermark in this video!')

    except Exception as e:
        print(e)
        print('failed to remove the watermark')
    return video_dst

def wm_video2(video_path): 
    '''''
    detect watermark in one  of the video frames
    remove watermark from all of frames and save to a new video
    '''''
    try:
        vid = imageio.get_reader(video_path,'ffmpeg')
        L = vid.get_length()        
        num = int(L/2)
        print("select %d%s frame for watermark detection:"%(num,'th'))
        image = vid.get_data(num)
        h = image.shape[0]
        w = image.shape[1]
        h_tmp = int(h/3)
        image_detec = image[0:h_tmp,:,:]
        image_np_expanded = np.expand_dims(image_detec, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        image_np,class_name,box_map = vis_util.visualize_boxes_and_labels_on_image_array(
                image_detec, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                category_index, use_normalized_coordinates=True, line_thickness=3)
        if len(class_name)>0:
            print('begain remove watermark ',class_name)
            fps = vid.get_meta_data()['fps']
            video_dst = video_path.split('/')[-1][0:-4] + '-3' + '.mp4' #where to save the new video
            writer = imageio.get_writer(video_dst, fps=fps, macro_block_size=None)
            #for num,im in enumerate(vid):  
            for i in range(0,L,5):              
                im = vid.get_data(i)
                im1 = vid.get_data(i+1)
                im2 = vid.get_data(i+2)
                im3 = vid.get_data(i+3)
                im4 = vid.get_data(i+4)
                im_tmp = im[0:h_tmp,:,:]
                wm_rec,rec = wm_remove2(im_tmp,box_map)
                im[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec[rec[0]:rec[1],rec[2]:rec[3],:]
                im1[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec[rec[0]:rec[1],rec[2]:rec[3],:]
                im2[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec[rec[0]:rec[1],rec[2]:rec[3],:]
                im3[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec[rec[0]:rec[1],rec[2]:rec[3],:]
                im4[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec[rec[0]:rec[1],rec[2]:rec[3],:]
                writer.append_data(im)
                writer.append_data(im1)
                writer.append_data(im2)
                writer.append_data(im3)
                writer.append_data(im4)
            writer.close()
        else:
            print('there is no watermark in this video!')

    except Exception as e:
        print(e)
        print('something wrong when remove the watermark')
        
def wm_image(image_path):
    '''''
    detect the watermark in a image and remove it.
    '''''
    try:
        image = imageio.imread(image_path)
        h = image.shape[0]
        w = image.shape[1]
        h_tmp = int(h/3)
        im_detec = image[0:h_tmp,:,:]
        image_np_expanded = np.expand_dims(im_detec, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        image_np,class_name,box_map = vis_util.visualize_boxes_and_labels_on_image_array(
                im_detec, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                category_index, use_normalized_coordinates=True, line_thickness=3)
        if len(class_name)>0:
            print('begain remove watermark ',class_name)
            image_name, ext = os.path.splitext(image_path)
            image_dst = image_name + '-2' + '.jpg' #where to save the new image 
            for box, color in box_map.items():
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin*h_tmp)
                xmin = int(xmin*w)
                ymax = int(ymax*h_tmp)
                xmax = int(xmax*w)
                wm_rec,rec = wm_remove(im_detec,ymin,xmin,ymax,xmax)
                im[rec[0]:rec[1],rec[2]:rec[3],:] = wm_rec
            imageio.imwrite(image_dst, image)

    except Exception as e:
        print(e)
        print('failed to remove the watermark')

def wm_remove_video(video_url):
    filename = video_url.split('/')[-1]
    try:
        down2file(video_url,filename,folder=IMAGE_FOLDER)
    except Exception as e:
        print('failed to download the video!')
    video_path = IMAGE_FOLDER + '/' + filename
    video_dst = wm_video2(video_path)
    try:
        os.remove(video_path)
        os.remove(video_dst)
    except Exception as e:
        pass
    return True,video_dst
    
if __name__ == '__main__':
    video_path = 'vvvvvv.mp4'
    wm_video2(video_path)
    video_url = 'http://flv3.bn.netease.com/videolib3/1801/31/JlUxq3843/SD/JlUxq3843-mobile.mp4'
    wm_remove_video(video_url)
    image_path = 'copy/haokan/1.jpg'
    wm_image(image_path)
