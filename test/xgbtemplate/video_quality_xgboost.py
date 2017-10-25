
# coding: utf-8

# In[54]:

import os
import numpy as np
import json
import pandas as pd

import sklearn
import xgboost as xgb
from sklearn import cross_validation
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ## 数据加载

# In[352]:

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
        
    org_feature['norm_resolution_value'] = (640.0 * 368) / (org_feature['video_height'] * org_feature['video_width']) * org_feature['max_resolution_value']
    org_feature['norm_mean_resolution'] = (640.0 * 368) / (org_feature['video_height'] * org_feature['video_width']) * org_feature['max_resolution_value']

    return org_feature
        
        


# In[353]:

data_base = "/data/hzwangjian1/videoquality"
positive_data_path = os.path.join(data_base, 'video_feature_positive')
negative_data_path = os.path.join(data_base, 'video_feature_negative')

positive_data = []
for line in open(positive_data_path, 'r'):
    feature = json.loads(line)
    feature.update({'label':1})
    positive_data.append(feature_process(feature))

print("size of positive_data:" + str(len(positive_data)))
    
negative_data = []
for line in open(negative_data_path, 'r'):
    feature = json.loads(line)
    feature.update({'label': 0})
    negative_data.append(feature_process(feature))
    
print("size of negative_data:" + str(len(negative_data)))



# In[354]:

total_data = pd.DataFrame(positive_data + negative_data)
total_data.head()


# In[355]:

total_data.info()
# total_data.to_csv('/data/hzwangjian1/videoquality/data_format.csv', sep='\t')


# In[293]:

total_data['label'].plot()


# In[374]:

total_data_copy = total_data.copy()
final_y = total_data_copy.pop('label')
final_x = total_data_copy.drop(['pic_url','big_image_url','category','quality','source_title','tid_level','tid_score','interests','doc_id','definition','title','m3u8HdUrl','m3u8SdUrl','m3u8ShdUrl','mp4HdUrl','mp4SdUrl','mp4ShdUrl','mp4_url','hdUrl','sdUrl','shdUrl','duration','video_duration','audio_duration','video_time_base','video_nb_frames','video_r_frame_rate','no_audio','size','size_per_pix','location_type','audio_nb_frames','video_avg_frame_rate','video_level','video_bit_rate','audio_bit_rate','bit_rate','keyframe_path','root_category','max_resolution_value','norm_resolution_value'], axis=1)


# In[375]:

final_x.info()


# In[303]:

final_x.info()


# In[386]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_x, final_y, test_size=0.33, random_state=42)
X_train.head()
weight = [2 if i==0 else 1 for i in y_train.tolist()]


# In[387]:

y_train.head()


# ## Xgboost进行数据重要性分析

# In[388]:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
xgdmat=xgb.DMatrix(X_train,y_train, weight=weight)
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':5,'min_child_weight':1,'objective':'binary:logistic','eval_metric':'logloss','n_estimators':100}
our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':5,'min_child_weight':1,'objective':'binary:logistic','n_estimators':100}
#our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:linear','n_estimators':100}
final_gb=xgb.train(our_params,xgdmat,num_boost_round=10)
final_gb.save_model(os.path.join(data_base, 'xgb_model'))

tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb.predict(tesdmat)
for i in range(400):
#     if y_test.iloc[i] == 0:
    print("{}\t{}".format(y_pred[i], y_test.iloc[i]))
y_pred_label = y_pred >= 0.5
    
import math
testScore=math.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:" + str(testScore))
accuracy_score_value = accuracy_score(y_test, y_pred_label)
print("accuracy_score:" + str(accuracy_score_value))
print(classification_report(y_test, y_pred_label))


# In[390]:

xgb.plot_importance(final_gb)


# In[ ]:



