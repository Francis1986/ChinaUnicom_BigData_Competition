#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from data_load import data_load
from fea_gen import fea_gen
from model import train_lgb
from sklearn.model_selection import KFold
from glob import glob
import lightgbm as lgb
import xgboost as xgb

#初始参数设置
Train_On = True
uptime_wash_on = True
one_hot_on = True
Submit_On = True
#信息泄露1和信息泄露2的格挡日期
Feb_mon = pd.to_datetime('20180215')
Mar_mon = pd.to_datetime('20180315')
#所有预测日期均在20120522到20180331之间
limit_date_end = pd.to_datetime('20180331')
limit_date_start = pd.to_datetime('20120522')
cate_col = [
        'CUST_SEX','IS_ACCT','USER_STATUS',
        'IS_CARD','IS_AGREE','IS_ZNZD','last_year_1','last_year_2',
        'AREA_ID',
        'CHANNEL_TYPE','CONSTELLATION_DESC'
    ]
drop_col = ['USER_ID','CUST_ID','DEVICE_NUMBER','MSISDN','CHNL_KIND_ID','gap_label','HIS_UP_TIME1','HIS_FACTORY_DESC1','HIS_TERM_DESC1','UP_TIME','INNET_DATE','ans_13','ans_23','Jan_uptime1','Feb_uptime1']
#特征工程
#原始数据导入初步处理
data = data_load()
#特征生成
fea_train,fea_test = fea_gen(data=data,one_hot_col=cate_col,uptime_wash_on=uptime_wash_on,one_hot_on=one_hot_on)



#训练数据，使用8折交叉验证
print ('开始训练数据，请稍后... ...')
if Train_On:
    train = fea_train.drop(drop_col,axis=1)
    #train[cate_col] = train[cate_col].astype('category')
    train_label_gap = fea_train['gap_label']
    train_label_date = fea_train[['HIS_UP_TIME1','UP_TIME']]
    kf = KFold(n_splits=8,shuffle=True,random_state=0)
    mse_list = []
    y_df_con = []
    fmp_list = []
    i=1
    for train_index,test_index in kf.split(np.arange(0,len(fea_train))):
        x_train = train.loc[train_index]
        y_train_gap = train_label_gap.loc[train_index]
        x_test = train.loc[test_index]
        y_test_gap = train_label_gap.loc[test_index]
        #进模型进行训练，输出模型
        model_lgb = train_lgb(x_train,y_train_gap,x_test,y_test_gap)
        model_lgb.save_model('./model/model_lgb_{}.csv'.format(i))#保存模型
        print ('模型保存完成！')
        #保存特征重要性
        fmp =  pd.DataFrame({'imp'+str(i):model_lgb.feature_importance()},index=list(train.columns))
        fmp_list.append(fmp)

        #线下预测
        y_pred_gap = model_lgb.predict(x_test,num_iteration=model_lgb.best_iteration)

        y_df = fea_train[['MSISDN','HIS_UP_TIME1','HIS_FACTORY_DESC1','HIS_TERM_DESC1','INNET_DATE','UP_TIME','Jan_uptime1','Feb_uptime1','1v3','2v3','11v22','21v32','ans_13','ans_23']].loc[test_index].reset_index(drop=True)
        y_df['y_pred_gap']=np.round(y_pred_gap)
        y_df['pred_date'] = y_df['HIS_UP_TIME1']+pd.to_timedelta(y_df['y_pred_gap'],unit='d')
        #所有预测日期均不超过20180331
        y_df['limit_date_end'] = limit_date_end #20180331
        y_df['pred_date_f'] = y_df[['pred_date','limit_date_end']].min(axis=1)
        #处理信息泄露
        y_df['pred_date_ff'] = y_df['pred_date_f']
        #y_df.loc[y_df['HIS_UP_TIME1']<y_df['INNET_DATE'],'pred_date_ff']=y_df.loc[y_df['HIS_UP_TIME1']<y_df['INNET_DATE'],'INNET_DATE']
        #信息泄露1:表1 1月count小于3月count的，uptime都在2018年2月之后，表1 2月count小于3月count的，uptime都在2018年3月之后,1v3,2v3
        #信息泄露2:表1 1月的uptime1等于2月的uptime2的3月换机时间一定在2018年2月或3月，2月的uptime1等于3月的uptime2,3月换机时间一定>是2018年3月11v22,21v32
        #y_df = y_df.merge(train_up,on='MSISDN',how='left')
        y_df['limit_date_s'] = limit_date_start #20120522
        y_df.loc[y_df['1v3']==1,'limit_date_s'] = Feb_mon
        y_df.loc[y_df['11v22']==1,'limit_date_s'] = Feb_mon
        y_df.loc[y_df['2v3']==1,'limit_date_s'] = Mar_mon
        y_df.loc[y_df['21v32']==1,'limit_date_s'] = Mar_mon
        y_df['pred_date_ff'] = y_df[['pred_date_ff','limit_date_s']].max(axis=1)
        #信息泄露3表1 1月2月 与3月的错位相当的，可以直接填充换机日期答案
        y_df.loc[y_df['ans_13']==1,'pred_date_ff'] = y_df.loc[y_df['ans_13']==1,'Jan_uptime1']
        y_df.loc[y_df['ans_23']==1,'pred_date_ff'] = y_df.loc[y_df['ans_23']==1,'Feb_uptime1']

        y_df['diff'] = (y_df['UP_TIME']-y_df['pred_date_ff']).dt.days
        y_df_con.append(y_df)
        #计算MSE
        mse = (y_df['diff'].values**2).sum()/len(y_pred_gap)
        print ('第{}个mse:{}'.format(i,mse))
        i+=1
        mse_list.append(mse)
    print ('mse_list:{}'.format(mse_list))
    print ('mse均值为:{}'.format(np.mean(mse_list)))
    y_df = pd.concat(y_df_con,axis=0,sort=False)
    fmp = pd.concat(fmp_list,axis=1,sort=False)
    fmp['mean_score'] = fmp.mean(axis=1)
    fmp.sort_values('mean_score',ascending=False).to_csv('./fmp/fmp.csv')
    y_df.to_csv('y_df.csv',index=None)
    print ('训练完毕！模型已生成！')

if Submit_On:
    print ('开始预测测试集数据，请稍后... ...')
    y_pred_df = fea_test[['USER_ID','MSISDN','HIS_UP_TIME1','INNET_DATE','1v3','2v3','11v22','21v32','ans_13','ans_23','Jan_uptime1','Feb_uptime1']].reset_index(drop=True)
    test = fea_test.drop(drop_col,axis=1)
    print ('测试集数量:',len(test))
    #离散特征转化成category格式进模型
    #test[cate_col] = test[cate_col].astype('category')
    #获取模型列表
    model_list = glob('./model/model_lgb*')
    print (model_list)
    pred_gap_col_list=[]
    for i,model in enumerate(model_list):
        model_lgb = lgb.Booster(model_file=model)
        y_pred_gap = model_lgb.predict(test,num_iteration=model_lgb.best_iteration)
        y_pred_df['y_pred_gap{}'.format(i)] = y_pred_gap
        pred_gap_col_list.append('y_pred_gap{}'.format(i))
    print ('测试集预测完成，开始保存结果... ...')
    y_pred_df['y_pred_gap_f']=np.round(y_pred_df[pred_gap_col_list].mean(axis=1))
    y_pred_df['pred_date'] = y_pred_df['HIS_UP_TIME1']+pd.to_timedelta(y_pred_df['y_pred_gap_f'],unit='d')
    #所有预测日期均在20180331之前
    y_pred_df['limit_date_end'] = limit_date_end #20180331
    y_pred_df['pred_date_f'] = y_pred_df[['pred_date','limit_date_end']].min(axis=1)

    #处理信息泄露
    #信息泄露1:表1 1月count小于3月count的，uptime都在2018年2月之后，表1 2月count小于3月count的，uptime都在2018年3月之后
    #信息泄露2: 表1 1月的uptime1等于2月的uptime2的3月换机时间一定在2018年2月或3月，2月的uptime1等于3月的uptime2,3月换机时间一定>是2018年3月
    y_pred_df['pred_date_ff'] = y_pred_df['pred_date_f']
    y_pred_df['limit_date_s'] = limit_date_start #20120522
    y_pred_df.loc[y_pred_df['1v3']==1,'limit_date_s'] = Feb_mon
    y_pred_df.loc[y_pred_df['11v22']==1,'limit_date_s'] = Feb_mon
    y_pred_df.loc[y_pred_df['2v3']==1,'limit_date_s'] = Mar_mon
    y_pred_df.loc[y_pred_df['21v32']==1,'limit_date_s'] = Mar_mon
    y_pred_df['pred_date_ff'] = y_pred_df[['pred_date_ff','limit_date_s']].max(axis=1)
    #信息泄露3：表1 1月2月 与3月的错位相当的，可以直接填充换机日期答案, 但是数量只有40个
    #print (y_pred_df[['pred_date_ff','Jan_uptime1','Feb_uptime1']])
    y_pred_df.loc[y_pred_df['ans_13']==1,'pred_date_ff'] = y_pred_df.loc[y_pred_df['ans_13']==1,'Jan_uptime1']
    y_pred_df.loc[y_pred_df['ans_23']==1,'pred_date_ff'] = y_pred_df.loc[y_pred_df['ans_23']==1,'Feb_uptime1']
    #将日期修改成要求的格式 i
    y_pred_df['pred_date_sub']=y_pred_df['pred_date_ff'].astype('str').apply(lambda x:x[:4]+x[5:7]+x[8:])
    result = y_pred_df[['USER_ID','pred_date_sub']].to_csv('./result/submit_lgb.csv',index=None,header=None)
    print('结果保存完毕！')