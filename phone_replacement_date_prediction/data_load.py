#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

def data_load():
    mon_1 = 201801
    mon_2 = 201802
    mon_3 = 201803
    print ('开始读取原始数据，并进行简单加工...')
    path = './data/'
    par = dict(delimiter='|',encoding='utf-8',header=None)
    #读取数据时直接对日期特征进行格式化处理，转化成datetime格式
    dates = [
        2,10,19,28,37,46
     ]
    data = {}
    #读取训练集
    tr1 = pd.read_table('{}Contest8_Training_1.txt'.format(path),**par,parse_dates=dates)
    tr2 = pd.read_table('{}Contest8_Training_2.txt'.format(path),**par)
    tr3 = pd.read_table('{}Contest8_Training_3.txt'.format(path),**par,parse_dates=[16])
    tr4 = pd.read_table('{}Contest8_Training_4.txt'.format(path),**par)
    #读取测试集
    te1 = pd.read_table('{}Contest8_Backups_1.txt'.format(path),**par,parse_dates=dates)
    te2 = pd.read_table('{}Contest8_Backups_2.txt'.format(path),**par)
    te3 = pd.read_table('{}Contest8_Backups_3.txt'.format(path),**par,parse_dates=[16])
    te4 = pd.read_table('{}Contest8_Backups_4.txt'.format(path),**par)


    column_1 = [
        'MONTH_ID','MSISDN','UP_TIME','HIS_IMEI1','HIS_IMSI1','HIS_FACTORY_ID1','HIS_TERM_ID1','HIS_FACTORY_DESC1',
        'HIS_TERM_DESC1','HIS_CITY_NO1','HIS_UP_TIME1','HIS_IS_COPY1','HIS_IMEI2','HIS_IMSI2','HIS_FACTORY_ID2',
        'HIS_TERM_ID2','HIS_FACTORY_DESC2','HIS_TERM_DESC2','HIS_CITY_NO2','HIS_UP_TIME2','HIS_IS_COPY2','HIS_IMEI3',
        'HIS_IMSI3','HIS_FACTORY_ID3','HIS_TERM_ID3','HIS_FACTORY_DESC3','HIS_TERM_DESC3','HIS_CITY_NO3','HIS_UP_TIME3',
        'HIS_IS_COPY3','HIS_IMEI4','HIS_IMSI4','HIS_FACTORY_ID4','HIS_TERM_ID4','HIS_FACTORY_DESC4','HIS_TERM_DESC4',
        'HIS_CITY_NO4','HIS_UP_TIME4','HIS_IS_COPY4','HIS_IMEI5','HIS_IMSI5','HIS_FACTORY_ID5','HIS_TERM_ID5',
        'HIS_FACTORY_DESC5','HIS_TERM_DESC5','HIS_CITY_NO5','HIS_UP_TIME5','HIS_IS_COPY5'
    ]
    column_2 = [
        'MONTH_ID','USER_ID','DEVICE_NUMBER','PRODUCT_CLASS','IS_CARD','IS_ADD','ACCT_FEE','IS_ZNZD','IS_AGREE'
    ]
    column_3 = [
        'MONTH_ID','USER_ID','CUST_ID','CUST_TYPE','USECUST_ID','SERVICE_TYPE','BRAND_ID','USER_DIFF_CODE',
        'DEVICE_NUMBER','NET_TYPE_CBSS','SCORE_VALUE','CREDIT_CLASS','BASIC_CREDIT_VALUE','CREDIT_VALUE',
        'IS_ACCT','PAY_MODE','INNET_DATE','OPER_DATE','OPEN_MODE','USER_STATUS','USER_STATUS_CBSS','CHANNEL_ID',
        'CHANNEL_TYPE','INNET_METHOD','IS_INNET','INNET_MONTHS'
    ]
    column_4 = [
        'MONTH_ID','CUST_ID','IS_INNET','CHNL_ID','CHNL_KIND_ID','AREA_ID','PAY_MODE',
        'CUST_SEX','CERT_AGE','CONSTELLATION_DESC'
    ]

    tr1.columns = te1.columns = column_1
    tr2.columns = te2.columns = column_2
    tr3.columns = te3.columns = column_3
    tr4.columns = te4.columns = column_4

    #对表1的count进行标识,为识别信息泄露做准备
    t1 = pd.concat([tr1,te1],axis=0,sort=False)
    #按日期筛选样本
    t1m1 = t1.loc[t1['MONTH_ID']==mon_1,:]
    t1m2 = t1.loc[t1['MONTH_ID']==mon_2,:]
    t1m3 = t1.loc[t1['MONTH_ID']==mon_3,:]

    t1m1['utc_m1'] = t1m1[['HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5']].count(axis=1)
    t1m2['utc_m2'] = t1m2[['HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5']].count(axis=1)
    t1m3['utc_m3'] = t1m3[['HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5']].count(axis=1)

    t1m1 = t1m1[['MSISDN','HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5','utc_m1']]
    t1m1['Jan_uptime1'] = t1m1['HIS_UP_TIME1']
    t1m1['Jan_uptime2'] = t1m1['HIS_UP_TIME2']
    t1m1['Jan_uptime3'] = t1m1['HIS_UP_TIME3']
    t1m1['Jan_uptime4'] = t1m1['HIS_UP_TIME4']
    t1m1['Jan_uptime5'] = t1m1['HIS_UP_TIME5']
    Jan_uptime =['Jan_uptime1','Jan_uptime2','Jan_uptime3','Jan_uptime4','Jan_uptime5']


    t1m2 = t1m2[['MSISDN','HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5','utc_m2']]
    t1m2['Feb_uptime1'] = t1m2['HIS_UP_TIME1']
    t1m2['Feb_uptime2'] = t1m2['HIS_UP_TIME2']
    t1m2['Feb_uptime3'] = t1m2['HIS_UP_TIME3']
    t1m2['Feb_uptime4'] = t1m2['HIS_UP_TIME4']
    t1m2['Feb_uptime5'] = t1m2['HIS_UP_TIME5']
    Feb_uptime =['Feb_uptime1','Feb_uptime2','Feb_uptime3','Feb_uptime4','Feb_uptime5']


    t1m3 = t1m3.merge(t1m1[['MSISDN','utc_m1']+Jan_uptime],on='MSISDN',how='left')
    t1m3 = t1m3.merge(t1m2[['MSISDN','utc_m2']+Feb_uptime],on='MSISDN',how='left')
    #逻辑1：如果第一个月的[uptime2,uptime3,uptime4,uptime5]与第三个月[uptime1,uptime2,uptime3,uptime4]换机时间相等，那么第一个月的uptime1就是该用用户第三个月的答案
    #同理： 如果第二个月的[uptime2,uptime3,uptime4,uptime5]与第三个月[uptime1,uptime2,uptime3,uptime4]换机时间相等，那么第二个月的uptime1就是该用用户第三个月的答案
    t1m3['ans_13'] = 0
    t1m3['ans_23'] = 0

    t1m3.loc[t1m3['Jan_uptime2']==t1m3['HIS_UP_TIME1'],'ans_13']=1
    t1m3.loc[t1m3['Feb_uptime2']==t1m3['HIS_UP_TIME1'],'ans_23']=1
    #逻辑2：如果第一个月换机次数小于第三个月的换机次数，那么用户的换机时间一定是2018年2月或者3月
    #逻辑3：如果第二个月换机次数小于第三个月的换机次数，那么用户的换机时间一定是2018年3月
    t1m3['11v22']=0
    t1m3['21v32']=0
    t1m3['1v3']=0
    t1m3['2v3']=0
    t1m3.loc[t1m3['utc_m1']<t1m3['utc_m3'],'1v3'] = 1
    t1m3.loc[t1m3['utc_m2']<t1m3['utc_m3'],'2v3'] = 1
    t1m3.loc[t1m3['Jan_uptime1']==t1m3['Feb_uptime2'],'11v22'] = 1
    t1m3.loc[t1m3['Feb_uptime1']==t1m3['HIS_UP_TIME2'],'21v32'] = 1


    #用户账期只使用2月份的即可
    tr2 = tr2.loc[tr2['MONTH_ID']==mon_2,:].drop_duplicates('USER_ID')
    te2 = te2.loc[te2['MONTH_ID']==mon_2,:].drop_duplicates('USER_ID')

    #读取品牌更新表
    brand_up = pd.read_csv('{}brand_up.csv'.format(path),encoding='utf-8')
    phone_info = pd.read_csv('{}phone_info_v2.csv'.format(path),encoding='utf-8',parse_dates=['product_time'])
    data['t1'] = t1m3
    data['t2'] = pd.concat([tr2,te2],axis=0,sort=False)
    data['t3'] = pd.concat([tr3,te3],axis=0,sort=False)
    data['t4'] = pd.concat([tr4,te4],axis=0,sort=False)

    data['t1'] = data['t1'].merge(brand_up,on='HIS_FACTORY_DESC1',how='left')
    data['t1'] = data['t1'].merge(phone_info,on=['HIS_FACTORY_DESC1','HIS_TERM_DESC1'],how='left')
    #删除无效字段
    drop_col = {}
    drop_col['t1'] = []
    drop_col['t2'] = [
        'MONTH_ID','PRODUCT_CLASS','IS_ADD'
        ]
    drop_col['t3'] = [
        'MONTH_ID','DEVICE_NUMBER','CUST_TYPE','USECUST_ID','SERVICE_TYPE','BRAND_ID','USER_DIFF_CODE','NET_TYPE_CBSS',
        'CREDIT_CLASS','BASIC_CREDIT_VALUE','CREDIT_VALUE','PAY_MODE','OPER_DATE','OPEN_MODE','USER_STATUS_CBSS',
        'CHANNEL_ID','INNET_METHOD'
    ]
    drop_col['t4'] = [
        'MONTH_ID','IS_INNET','CHNL_ID','PAY_MODE'
    ]
    for key in data.keys():
        data[key].drop(drop_col[key],axis=1,inplace=True)
    print ('原始数据读取完毕！')
    return data