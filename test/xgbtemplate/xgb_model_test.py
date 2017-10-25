
# coding: utf-8

# In[2]:

import os
import numpy as np
import json
import pandas as pd

import sklearn
import xgboost as xgb
from sklearn import cross_validation
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report


# In[58]:

## load xgb model
data_base = "/data/hzwangjian1/videoquality"
xgb_model = xgb.Booster({'nthread':10}) #init model
xgb_model.load_model(os.path.join(data_base, 'xgb_model')) # load data



# In[6]:

def feature_process(org_feature):
    if org_feature['location_type'] != 1:  
        org_feature['location_type'] = 0  ## 非头条，设置为0
        
    if org_feature['tid_level'] == 1 or org_feature['tid_level'] == 2: ## 稿源评级，优质次优的放一起
        org_feature.update({'tid_score':2})
    else:
        org_feature.update({'tid_score':1})
    
    if  'http' in org_feature['big_image_url']:
        org_feature.update({'contain_big_image':1})
    else:
        org_feature.update({'contain_big_image':0})
        
    definition = org_feature['definition'].lower().strip()
    if definition == 'shd':
        org_feature.update({'definition_score':3})
    elif definition == 'hd':
        org_feature.update({'definition_score':2})
    else:
        org_feature.update({'definition_score':1})
    return org_feature


# In[59]:

data_path = os.path.join(data_base, 'labeled_dataset_feature')

feature_set = []
for line in open(data_path, 'r'):
    feature = json.loads(line)
    feature_set.append(feature_process(feature))

print("size of feature_set:" + str(len(feature_set)))

total_data = pd.DataFrame(feature_set)
total_data.head()


# In[39]:

total_data.info()


# In[60]:

feature_result = []
for line in feature_set:
    data_feature = [line]
    data_frame = pd.DataFrame(data_feature)
    final_y = data_frame.pop('label')
    final_x = data_frame.drop(['pic_url','big_image_url','category','quality','source_title','tid_level','tid_score','interests','doc_id','definition','title','m3u8HdUrl','m3u8SdUrl','m3u8ShdUrl','mp4HdUrl','mp4SdUrl','mp4ShdUrl','mp4_url','hdUrl','sdUrl','shdUrl','duration','video_duration','audio_duration','video_time_base','video_nb_frames','video_r_frame_rate','no_audio','size','size_per_pix','location_type','audio_nb_frames','video_avg_frame_rate','video_level','video_bit_rate','audio_bit_rate','bit_rate'], axis=1)
    tesdmat=xgb.DMatrix(final_x)
    y_pred=xgb_model.predict(tesdmat)
    line.update({'y_pred': y_pred[0]})
    feature_result.append(line)

result_data = pd.DataFrame(feature_result)
result_data.to_csv(os.path.join(data_base, 'labeled_dataset_result_full.csv'), sep='\t')
    


# In[57]:

feature_result = []
key_list = ['pic_url','big_image_url','category','quality','source_title','tid_level','tid_score','interests','title','m3u8HdUrl','m3u8SdUrl','m3u8ShdUrl','mp4HdUrl','mp4SdUrl','mp4ShdUrl','mp4_url','hdUrl','sdUrl','shdUrl','duration','video_duration','audio_duration','video_time_base','video_nb_frames','video_r_frame_rate','no_audio','size','size_per_pix','location_type','audio_nb_frames','video_avg_frame_rate','video_level','video_bit_rate','audio_bit_rate','bit_rate']
for line in feature_set:
    data_feature = [line]
    data_frame = pd.DataFrame(data_feature)
    final_y = data_frame.pop('label')
    final_x = data_frame.drop(['pic_url','big_image_url','category','quality','source_title','tid_level','tid_score','interests','doc_id','definition','title','m3u8HdUrl','m3u8SdUrl','m3u8ShdUrl','mp4HdUrl','mp4SdUrl','mp4ShdUrl','mp4_url','hdUrl','sdUrl','shdUrl','duration','video_duration','audio_duration','video_time_base','video_nb_frames','video_r_frame_rate','no_audio','size','size_per_pix','location_type','audio_nb_frames','video_avg_frame_rate','video_level','video_bit_rate','audio_bit_rate','bit_rate'], axis=1)
    tesdmat=xgb.DMatrix(final_x)
    y_pred=xgb_model.predict(tesdmat)
    line.update({'y_pred': y_pred[0]})
    for key in key_list:
        line.pop(key)
    feature_result.append(line)

result_data = pd.DataFrame(feature_result)
result_data.to_csv(os.path.join(data_base, 'labeled_dataset_result.csv'), sep='\t')

