#!/usr/bin/python
# -*- coding:utf-8 -*-
from data_load import data_load
from fea_gen import fea_gen,one_hot
from sklearn.model_selection import KFold
from model import lgb_predict,xgb_predict
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time

user_fea_exist = True
Train = True
Submit =  False
data_path = './data/'
train_path = './feature/user_fea_train_all_curend_xgb.csv'
test_path = './feature/user_fea_test_all_curend_xgb.csv'


if not user_fea_exist:
    path = './data/'
    data = data_load(data_path)
    time_1 = time.clock()
    time_gap_set = [201802,201801,201712]
    user_fea_train,user_fea_test = fea_gen(data,time_gap_set,one_hot_on=True)
    time_2 = time.clock()
    print ('feature generation duration is',time_2-time_1,'s')
    print ('user_fea_train length',len(user_fea_train))
    print ('user_fea_test length',len(user_fea_test))
    print ('column nums:',len(user_fea_train.columns))
    user_fea_train.to_csv(train_path,index=None)
    user_fea_test.to_csv(test_path,index=None)
    data_train = []
    data_test = []
    # user_fea_train = []
    # user_fea_test = []

if Train:
    print ('loading features,hold on...')
    train_data = pd.read_csv(train_path)
    #train_data = user_fea_train
    user_fea_train=[]
    print ('feature loading is complete!')
    data = train_data.drop(['USER_ID','CUST_ID','IS_LOST'],axis=1)
    # data[cate_columns]=data[cate_columns].astype('category')
    label = train_data['IS_LOST']
    proba_th = 0.5
    param = {
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':6,
    'min_child_weight':0.001,
    'eta':0.05,
    'colsample_bytree': 0.8,
    'subsample':0.8
    }
    kf = KFold(n_splits=4,shuffle=True,random_state=0)
    score_list = []
    gbm_list = []
    tn_list = []
    for train_index,test_index in kf.split(np.arange(0,len(data))):
        x_train = data.loc[train_index]
        y_train = label.loc[train_index]
        x_test = data.loc[test_index]
        y_test = label.loc[test_index]
        print ('train_len:',len(x_train))
        print ('test_len:',len(x_test))
        y_pred,gbm = xgb_predict(x_train,y_train,x_test,y_test,param)
        gbm_list.append(gbm)
        y_pred[y_pred >= proba_th] = 1
        y_pred[y_pred < proba_th] = 0
        t_length = len(y_pred[y_pred==1])
        n_length = len(y_pred[y_pred==0])
        score = f1_score(y_test,y_pred)
        print (score)
        score_list.append(score)
        tn_list.append({0:n_length,1:t_length})
    print ('score list:',score_list)
    print ('mean score:',np.mean(score_list))
    print ('tn_list:',tn_list)

if Submit:
   print ('generating submission file,hold on please...')
   test_data = pd.read_csv(test_path)
   #test_data = user_fea_test
   user_fea_test = []
   x_test = test_data.drop(['USER_ID','CUST_ID','IS_LOST'],axis=1)
   print ('test length:',len(x_test))
   #x_test[cate_columns]= x_test[cate_columns].astype('category')
   y_pred_df = pd.DataFrame(index=test_data['USER_ID'])
   y_pred_df['USER_ID']=y_pred_df.index
   for i,gbm in enumerate(gbm_list):
       xgb_test = xgb.DMatrix(x_test.values)
       y_pred = gbm.predict(xgb_test)
       y_pred_df.loc[:,str(i)+'col']=y_pred
   y_pred_df['final_proba'] =  y_pred_df.mean(axis=1)
   y_pred_df['IS_LOST']=''
   y_pred_df.loc[y_pred_df['final_proba']>=proba_th,'IS_LOST']=1
   y_pred_df.loc[y_pred_df['final_proba']<proba_th,'IS_LOST']=0
   y_pred_df.to_csv('r_xgb_eval.csv',index=None)
   print ('result_length:',len(y_pred_df))
   submit = y_pred_df[['USER_ID','IS_LOST']]
   submit.to_csv('./result/2_网研1队_20180822_curend_xgb_eval.csv',index=None,header=None)
   print ('submission file is generated,please check!')