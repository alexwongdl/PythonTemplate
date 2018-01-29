
# coding: utf-8

# In[1]:

import os
import numpy as np
import json
import pandas as pd

import sklearn
import xgboost as xgb
from sklearn import cross_validation
%matplotlib inline
import matplotlib.pyplot as plt


# In[2]:

data_path = '/data/hzwangjian1/image_quality/'
data_0 = os.path.join(data_path, '0_feature_resize')
data_1 = os.path.join(data_path, '1_feature_resize')
data_2 = os.path.join(data_path, '2_feature_resize')


# ## load data

# In[3]:

def load_feature(feature_file):
    features = []
    with open(feature_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            features.append(json.loads(line))
    return features


# In[4]:

features_0 = load_feature(data_0)
features_1 = load_feature(data_1)
features_2 = load_feature(data_2)

print('length of features_0:{}'.format(len(features_0)))
print('length of features_1:{}'.format(len(features_1)))
print('length of features_2:{}'.format(len(features_2)))


# In[5]:

train_feature_0 = features_0[0:5237] 
test_feature_0 = features_0[5237:] 
train_feature_1 = features_1[0:1703]
test_feature_1 = features_1[1703:] 
train_feature_2 = features_2[0:3528] 
test_feature_2 = features_2[3528:] 

print('len of train_feature_0:{}'.format(len(train_feature_0)))
print('len of train_feature_1:{}'.format(len(train_feature_1)))
print('len of train_feature_2:{}'.format(len(train_feature_2)))
print('len of test_feature_0:{}'.format(len(test_feature_0)))
print('len of test_feature_1:{}'.format(len(test_feature_1)))
print('len of test_feature_2:{}'.format(len(test_feature_2)))


# In[6]:

total_data = pd.DataFrame(train_feature_0 + train_feature_1 + train_feature_2)
total_data.head()

total_test_data = pd.DataFrame(test_feature_0 + test_feature_2)
total_test_data.head()


# In[7]:

total_data.info()


# In[8]:

total_data_copy = total_data.copy()
final_y = total_data_copy.pop('label')
final_x = total_data_copy.drop(['image_path'], axis=1)
final_x.info()


# In[9]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_x, final_y, test_size=0.1, random_state=42)
X_train.head()


# In[10]:

print(X_train.shape)
print(X_test.shape)


# In[11]:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report


# ## 两个分类器

# ### 0/1一类，2单独一个类别

# In[23]:

total_data_copy = total_data.copy()
final_y = total_data_copy.pop('label')
final_x = total_data_copy.drop(['image_path'], axis=1)
final_x.info()


# In[24]:

final_y = final_y.replace(1, 0)
final_y = final_y.replace(2, 1)   # 2-->1 , 1-->0, 0-->0
X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_x, final_y, test_size=0.1, random_state=42)
X_train.head()


# In[25]:

weight = [5 if i==1 else 1 for i in y_train.tolist()]


# ## 加weight，used now

# # wight = 2

# In[15]:

xgdmat=xgb.DMatrix(X_train,label=y_train, weight=weight)
valid_mat=xgb.DMatrix(X_test,label=y_test)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'binary:logistic','eval_metric':'logloss','n_estimators':100}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:logistic','n_estimators':200,'silent':1,'verbose':1}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'multi:softmax','eval_metric':'mlogloss','n_estimators':200, 'num_class':3}
#our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:linear','n_estimators':100}
final_gb_01_2=xgb.train(our_params,xgdmat,num_boost_round=500, evals=[(xgdmat, 'x_train'),(valid_mat,'x_test')])
final_gb_01_2.save_model(os.path.join(data_path, 'xgb_image_quality_01_2_AW_resize'))

tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb_01_2.predict(tesdmat)
for i in range(100):
#     if y_test.iloc[i] == 0:
    print("{}\t{}".format(y_pred[i], y_test.iloc[i]))
y_pred_label = y_pred >= 0.5
    
import math
# testScore=math.sqrt(mean_squared_error(y_test,y_pred))
# print("RMSE:" + str(testScore))
accuracy_score_value = accuracy_score(y_test, y_pred_label)
print("accuracy_score:" + str(accuracy_score_value))
print(classification_report(y_test, y_pred_label))


# In[26]:

xgdmat=xgb.DMatrix(X_train,label=y_train, weight=weight)
valid_mat=xgb.DMatrix(X_test,label=y_test)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'binary:logistic','eval_metric':'logloss','n_estimators':100}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:logistic','n_estimators':200,'silent':1,'verbose':1}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'multi:softmax','eval_metric':'mlogloss','n_estimators':200, 'num_class':3}
#our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:linear','n_estimators':100}
final_gb_01_2=xgb.train(our_params,xgdmat,num_boost_round=500, evals=[(xgdmat, 'x_train'),(valid_mat,'x_test')])
final_gb_01_2.save_model(os.path.join(data_path, 'xgb_image_quality_01_2_AW_resize'))

tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb_01_2.predict(tesdmat)
for i in range(100):
#     if y_test.iloc[i] == 0:
    print("{}\t{}".format(y_pred[i], y_test.iloc[i]))
y_pred_label = y_pred >= 0.5
    
import math
# testScore=math.sqrt(mean_squared_error(y_test,y_pred))
# print("RMSE:" + str(testScore))
accuracy_score_value = accuracy_score(y_test, y_pred_label)
print("accuracy_score:" + str(accuracy_score_value))
print(classification_report(y_test, y_pred_label))


# In[27]:

xgb.plot_importance(final_gb_01_2)


# ### 2/1一类，0单独一个类别

# In[28]:

total_data_copy = total_data.copy()
final_y = total_data_copy.pop('label')
final_x = total_data_copy.drop(['image_path'], axis=1)
final_x.info()


# In[29]:

final_y = final_y.replace(2, 1)  # 2-->1 , 1-->1, 0-->0 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_x, final_y, test_size=0.1, random_state=42)
X_train.head()


# In[30]:

weight = [2 if i==1 else 1 for i in y_train.tolist()]


# In[31]:

xgdmat=xgb.DMatrix(X_train,label=y_train, weight=weight)
valid_mat=xgb.DMatrix(X_test,label=y_test)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'binary:logistic','eval_metric':'logloss','n_estimators':100}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:logistic','n_estimators':200,'silent':1,'verbose':1}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'multi:softmax','eval_metric':'mlogloss','n_estimators':200, 'num_class':3}
#our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:linear','n_estimators':100}
final_gb_0_12=xgb.train(our_params,xgdmat,num_boost_round=500, evals=[(xgdmat, 'x_train'),(valid_mat,'x_test')])
final_gb_0_12.save_model(os.path.join(data_path, 'xgb_image_quality_0_12_AW_resize'))

tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb_0_12.predict(tesdmat)
for i in range(100):
#     if y_test.iloc[i] == 0:
    print("{}\t{}".format(y_pred[i], y_test.iloc[i]))
y_pred_label = y_pred >= 0.5
    
import math
# testScore=math.sqrt(mean_squared_error(y_test,y_pred))
# print("RMSE:" + str(testScore))
accuracy_score_value = accuracy_score(y_test, y_pred_label)
print("accuracy_score:" + str(accuracy_score_value))
print(classification_report(y_test, y_pred_label))


# In[80]:

xgdmat=xgb.DMatrix(X_train,label=y_train, weight=weight)
valid_mat=xgb.DMatrix(X_test,label=y_test)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'binary:logistic','eval_metric':'logloss','n_estimators':100}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:logistic','n_estimators':200,'silent':1,'verbose':1}
# our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'multi:softmax','eval_metric':'mlogloss','n_estimators':200, 'num_class':3}
#our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'max_depth':6,'min_child_weight':1,'objective':'reg:linear','n_estimators':100}
final_gb_0_12=xgb.train(our_params,xgdmat,num_boost_round=500, evals=[(xgdmat, 'x_train'),(valid_mat,'x_test')])
final_gb_0_12.save_model(os.path.join(data_path, 'xgb_image_quality_0_12_AW_resize'))

tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb_0_12.predict(tesdmat)
for i in range(100):
#     if y_test.iloc[i] == 0:
    print("{}\t{}".format(y_pred[i], y_test.iloc[i]))
y_pred_label = y_pred >= 0.5
    
import math
# testScore=math.sqrt(mean_squared_error(y_test,y_pred))
# print("RMSE:" + str(testScore))
accuracy_score_value = accuracy_score(y_test, y_pred_label)
print("accuracy_score:" + str(accuracy_score_value))
print(classification_report(y_test, y_pred_label))


# In[32]:

xgb.plot_importance(final_gb_0_12)


# ## 模型融合测试

# In[33]:

total_test_data_copy = total_test_data.copy()
final_y_test = total_test_data_copy.pop('label')
final_x_test = total_test_data_copy.drop(['image_path'], axis=1)
final_x_test.info()


# In[34]:


print(final_x_test.shape)
print(final_y_test)


# In[35]:

tesdmat=xgb.DMatrix(final_x_test)
y_pred_01_2=final_gb_01_2.predict(tesdmat)
y_pred_01_2_label = y_pred_01_2 >= 0.5
y_pred_01_2_label


# In[36]:

y_pred_0_12 = final_gb_0_12.predict(tesdmat)
y_pred_0_12_label = y_pred_0_12 >= 0.5
y_pred_0_12_label


# In[37]:

predict_final = []
for i in range(len(y_pred_0_12_label)):
    class_label = -1
    if y_pred_01_2_label[i] and y_pred_0_12_label[i]: # 2和12
        class_label = 2
    elif y_pred_01_2_label[i] and not y_pred_0_12_label[i]:# 2和0
        class_label = 1
    elif not y_pred_01_2_label[i] and not y_pred_0_12_label[i]: # 01和0
        class_label = 0
    elif not y_pred_01_2_label[i] and y_pred_0_12_label[i]: # 01 和1
        class_label = 1
    predict_final.append(class_label)

predict_final


# In[38]:

label_compare = np.zeros(shape=(3,3))
for i in range(len(predict_final)): #final_y_test
    label_compare[final_y_test[i], predict_final[i]] += 1

print('\t\tpredict')
print('\t0\t1\t2\tsum')
print('0\t{}\t{}\t{}\t{}'.format(label_compare[0,0], label_compare[0,1], label_compare[0,2], sum(label_compare[0,:])))
print('1\t{}\t{}\t{}\t{}'.format(label_compare[1,0], label_compare[1,1], label_compare[1,2], sum(label_compare[1,:])))
print('2\t{}\t{}\t{}\t{}'.format(label_compare[2,0], label_compare[2,1], label_compare[2,2], sum(label_compare[2,:])))


# In[39]:

label_compare_ratio = np.zeros(shape=(3,3))
label_compare_row_sum = np.sum(label_compare,axis=1)
label_compare_ratio = label_compare / label_compare.sum(axis=1)[:,None]
label_compare_ratio


# In[ ]:



