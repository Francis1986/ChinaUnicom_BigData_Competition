#!/usr/bin/python
# -*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

def fea_gen(data,one_hot_col,uptime_wash_on=True,one_hot_on=False):
    print ('开始进行特征生成...')
    #构建表3和表4特征,用户信息表，关联表3与表4
    #表3的device_number直接关联表1的MSISDN会有缺失值，需要用表2的去关联
    #user_info的主键为USER_ID,user_info的device_NUMBER需要替换成表2的device_number
    user_info = data['t3'].merge(data['t4'],on='CUST_ID',how='left')
    #构建表2特征,用户出账表
    user_fee = data['t2']
    #构建表1特征
    data['t1'] = data['t1'].merge(data['t2'][['USER_ID','DEVICE_NUMBER']],left_on = 'MSISDN',right_on='DEVICE_NUMBER',how='left')
    user_update_gap = data['t1'].merge(data['t3'][['USER_ID','INNET_DATE']],on='USER_ID',how='left')
    if uptime_wash_on:
       time = ['HIS_UP_TIME1','HIS_UP_TIME2','HIS_UP_TIME3','HIS_UP_TIME4','HIS_UP_TIME5']
       user_update_gap['min_up_time'] = user_update_gap[time].min(axis=1)
       user_update_gap['max_up_time'] = user_update_gap[time].max(axis=1)
       user_update_gap['wash_up_time'] = 0
       user_update_gap.loc[(user_update_gap['min_up_time']<user_update_gap['INNET_DATE'])&(user_update_gap['max_up_time']>user_update_gap['INNET_DATE']),'wash_up_time']=1
       for col in time:
           user_update_gap.loc[(user_update_gap['wash_up_time']==1)&(user_update_gap[col]<user_update_gap['INNET_DATE']),col]=np.nan
    user_update_gap['is_12_repeat'] = 0
    user_update_gap['is_13_repeat'] = 0
    user_update_gap['is_14_repeat'] = 0
    user_update_gap['is_15_repeat'] = 0
    user_update_gap['is_repeat'] = 0
    user_update_gap.loc[user_update_gap['HIS_TERM_DESC1']==user_update_gap['HIS_TERM_DESC2'],['is_12_repeat','is_repeat']] = 1
    user_update_gap.loc[user_update_gap['HIS_TERM_DESC1']==user_update_gap['HIS_TERM_DESC3'],['is_13_repeat','is_repeat']] = 1
    user_update_gap.loc[user_update_gap['HIS_TERM_DESC1']==user_update_gap['HIS_TERM_DESC4'],['is_14_repeat','is_repeat']] = 1
    user_update_gap.loc[user_update_gap['HIS_TERM_DESC1']==user_update_gap['HIS_TERM_DESC5'],['is_15_repeat','is_repeat']] = 1
    user_update_gap['is_repeat_num'] = user_update_gap[['is_12_repeat','is_13_repeat','is_14_repeat','is_15_repeat']].sum(axis=1)
    user_update_gap['gap_now_uptime1'] = (pd.to_datetime('20180331')-user_update_gap['HIS_UP_TIME1']).dt.days
    user_update_gap['gap_now_product'] = (pd.to_datetime('20180331')-user_update_gap['product_time']).dt.days
    user_update_gap['up_gap_1_innet'] = (user_update_gap['HIS_UP_TIME1']-user_update_gap['INNET_DATE']).dt.days
    user_update_gap['up_gap_2_innet'] = (user_update_gap['HIS_UP_TIME2']-user_update_gap['INNET_DATE']).dt.days
    user_update_gap['up_gap_12'] = (user_update_gap['HIS_UP_TIME1']-user_update_gap['HIS_UP_TIME2']).dt.days
    user_update_gap['up_gap_23'] = (user_update_gap['HIS_UP_TIME2']-user_update_gap['HIS_UP_TIME3']).dt.days
   # user_update_gap['up_gap_34'] = (user_update_gap['HIS_UP_TIME3']-user_update_gap['HIS_UP_TIME4']).dt.days
   # user_update_gap['up_gap_45'] = (user_update_gap['HIS_UP_TIME4']-user_update_gap['HIS_UP_TIME5']).dt.days
    user_update_gap['last_year_1'] = (user_update_gap['HIS_UP_TIME1']).dt.year
    user_update_gap['last_year_2'] = (user_update_gap['HIS_UP_TIME2']).dt.year

    user_update_gap['gap_label'] = (user_update_gap['UP_TIME']-user_update_gap['HIS_UP_TIME1']).dt.days


    #缺失值平均数填充
    user_update_gap.fillna(
        {
         'up_gap_12':user_update_gap['up_gap_12'].mean(),
         'up_gap_23':user_update_gap['up_gap_23'].mean()
    #    'up_gap_34':user_update_gap['up_gap_34'].mean(),
    #    'up_gap_45':user_update_gap['up_gap_45'].mean()
        },
        inplace=True
    )
    gap_col = ['up_gap_12','up_gap_23']
    #用户历史换机间隔的最大值，最小值，平均值，中位数
    user_update_gap = user_update_gap.assign(
        up_gap_min=user_update_gap[gap_col].min(axis=1),
        up_gap_max=user_update_gap[gap_col].max(axis=1),
        up_gap_mean=user_update_gap[gap_col].mean(axis=1),
        up_gap_median=user_update_gap[gap_col].median(axis=1)
        )


    #排名topN的品牌进行onehot编码
    user_update_gap['is_unknown'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '**' else 0)
    user_update_gap['is_apple'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '苹果' else 0)
    user_update_gap['is_huawei'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '华为' or x=='荣耀' else 0)
    user_update_gap['is_xiaomi'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '小米' else 0)
    user_update_gap['is_vivo'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == 'vivo' else 0)
    user_update_gap['is_oppo'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == 'oppo' else 0)
    user_update_gap['is_samsung'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '三星' else 0)
    user_update_gap['is_meizu'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '魅族' else 0)
    user_update_gap['is_lenovo'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '联想' else 0)
    user_update_gap['is_leshi'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '乐视' else 0)
    user_update_gap['is_coolpad'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '酷派' else 0)
    user_update_gap['is_bbk'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '步步高' else 0)
    user_update_gap['is_Nokia'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '诺基亚' or x=='Nokia' else 0)
    user_update_gap['is_jinli'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '金立' else 0)
    user_update_gap['is_zte'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '中兴' else 0)
    user_update_gap['is_smartisan'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '锤子' else 0)
    user_update_gap['is_htc'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == 'HTC' else 0)
    user_update_gap['is_tianyu'] = user_update_gap['BRAND_UP'].apply(lambda x: 1 if x == '天语' else 0)

    user_gap_col_f = [
        'MSISDN','gap_now_uptime1',
        'gap_now_product',
        'up_gap_1_innet','up_gap_2_innet',
        'up_gap_12','up_gap_23','up_gap_min','up_gap_max','up_gap_mean','up_gap_median',
        'is_apple','is_huawei','is_xiaomi','is_vivo','is_oppo','is_samsung','is_meizu',
        'is_lenovo','is_leshi','is_coolpad','is_bbk','is_Nokia','is_jinli','is_zte','is_smartisan',
        'is_htc','is_tianyu','is_unknown','HIS_IS_COPY1',
        'is_12_repeat','is_13_repeat','is_14_repeat','is_15_repeat','is_repeat_num',
        'gap_label','last_year_1','last_year_2',
        '2Gratio','3Gratio','4Gratio','is_sp',
        'HIS_UP_TIME1','HIS_FACTORY_DESC1','HIS_TERM_DESC1','UP_TIME',
        'Jan_uptime1','Feb_uptime1','1v3','2v3','11v22','21v32','ans_13','ans_23'
        ]
    user_update_gap = user_update_gap[user_gap_col_f]
    #整理完成特征处理
    user_fea = user_fee.merge(user_info,on='USER_ID',how = 'left')
    user_fea = user_fea.merge(user_update_gap,left_on='DEVICE_NUMBER',right_on='MSISDN',how='left')


    #离散特征标签化处理
    cate_col = [
        'CHNL_KIND_ID','CUST_SEX','IS_ACCT','USER_STATUS',
        'IS_CARD','IS_ZNZD','IS_AGREE'
    ]
    le = LabelEncoder()
    for col in cate_col:
        user_fea[col] = le.fit_transform(user_fea[col])
    #AREA_ID中既有字母又有数字，LabelEncoder不支持，因此手动修改一下
    i=0
    for cate in user_fea['AREA_ID'].unique():
        user_fea.loc[user_fea['AREA_ID']==cate,'AREA_ID']=i
        i+=1
    i=0
    for cate in user_fea['CHANNEL_TYPE'].unique():
        user_fea.loc[user_fea['CHANNEL_TYPE']==cate,'CHANNEL_TYPE']=i
        i+=1
    i=0
    for cate in user_fea['CONSTELLATION_DESC'].unique():
        user_fea.loc[user_fea['CONSTELLATION_DESC']==cate,'CONSTELLATION_DESC']=i
        i+=1
    if one_hot_on:
       user_fea[one_hot_col] = user_fea[one_hot_col].astype('str')
       one_hot_fea = pd.get_dummies(user_fea[one_hot_col],prefix=one_hot_col)
       user_fea = pd.concat([user_fea.drop(one_hot_col,axis=1),one_hot_fea],axis=1,sort=False)
    #离散特征缺失值填充
    user_fea.fillna({'CERT_AGE':user_fea['CERT_AGE'].median(),'CONSTELLATION_DESC':-1},inplace=True)
    #训练集和测试集分解
    user_fea_train = user_fea.loc[~user_fea['UP_TIME'].isnull(),:]
    user_fea_test = user_fea.loc[user_fea['UP_TIME'].isnull(),:]
    print ('训练集样本数量：{}'.format(len(user_fea_train)))
    print ('测试集样本数量：{}'.format(len(user_fea_test)))
    print ('特征生成完毕！')
    return user_fea_train,user_fea_test