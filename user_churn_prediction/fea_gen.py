#!/usr/bin/python
# -*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder


def fea_gen(data,time_gap_set,one_hot_on):
    #销账新特征加入
    #销账截止账期-销账开始账期
    data['t5']['cycle_gap'] = \
    (pd.to_datetime(data['t5']['END_CYCLE_ID'].astype('str')+'01').dt.year-pd.to_datetime(data['t5']['START_CYCLE_ID'].astype('str')+'01').dt.year)*12+\
    (pd.to_datetime(data['t5']['END_CYCLE_ID'].astype('str')+'01').dt.month-pd.to_datetime(data['t5']['START_CYCLE_ID'].astype('str')+'01').dt.month)

    #2018年3月减去销账结束日期以及销账开始日期的值
    data['t5']['cur_end_gap'] = (2018 - pd.to_datetime(data['t5']['END_CYCLE_ID'].astype('str')+'01').dt.year)*12 + (3 - pd.to_datetime(data['t5']['END_CYCLE_ID'].astype('str')+'01').dt.month)
    data['t5']['cur_start_gap'] = (2018 - pd.to_datetime(data['t5']['START_CYCLE_ID'].astype('str')+'01').dt.year)*12 + (3 - pd.to_datetime(data['t5']['START_CYCLE_ID'].astype('str')+'01').dt.month)

    user_info_train = data['t1'].iloc[0:38903,:].merge(data['t4'].iloc[0:37994,:],on='CUST_ID',how='left').drop_duplicates(['USER_ID'],keep = 'first')
    user_info_test = data['t1'].iloc[38903:,:].merge(data['t4'].iloc[37994:,:],on='CUST_ID',how='left').drop_duplicates(['USER_ID'],keep = 'first')
    user_info = pd.concat([user_info_train,user_info_test],axis=0,sort=False)
    print ('user_info_train length:',len(user_info_train))
    print ('user_info_test length:',len(user_info_test))
    print ('user_info length:',len(user_info))
    #在网时长-天级
    user_info['INNET_DATE_gap'] = (pd.to_datetime('2018-03-01')-pd.to_datetime(user_info['INNET_DATE'].astype('str'))).dt.days
    user_info['OPER_DATE_gap'] = (pd.to_datetime('2018-03-01')-pd.to_datetime(user_info['OPER_DATE'].astype('str'))).dt.days
    user_info = user_info.drop(['INNET_DATE','OPER_DATE'],axis=1)
    #CERT_AGE deal
    user_info['CERT_AGE'] = user_info['CERT_AGE'].fillna(-1)
    age_bins = [-10,1,18,25,30,35,40,45,50,60,100]
    age_group = [-1,0,1,2,3,4,5,6,7,8]
    user_info['age_bins'] = pd.cut(user_info['CERT_AGE'],age_bins,labels = age_group)
    #user_info = user_info.drop('CERT_AGE',axis=1)
    user_info['age_bins'] = user_info['age_bins'].astype('int')
    print ('feature is generating... hold on please...')
    user_fea_concat = []
    time_gap_s = []
    #处理奇数偶数月金额
    data['t5']['oe_money'] = data['t5']['ODD_MONEY']+data['t5']['EVEN_MONEY']
    for time_gap in time_gap_set:
        user_traffic = pd.DataFrame(index=user_info['USER_ID'])
        user_traffic_nw = pd.DataFrame(index=user_info['USER_ID'])
        time_gap_s.append(time_gap)
        print (time_gap_s)
        print ('time_gap:',time_gap,'month')

        if len(time_gap_s)==1:
            #tr2 table
            user_traffic['TOTAL_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'mean'})
            user_traffic['BASE_RENT_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'mean'})
            user_traffic['DINNER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'mean'})
            user_traffic['FUNCATION_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'mean'})
            user_traffic['OTHER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'mean'})
            user_traffic['BASE_CALL_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'mean'})
            user_traffic['COUN_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'mean'})
            user_traffic['INTER_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'mean'})
            #Attention!!!lots values of GAT_LONG_LFEE are zeros only 30 not zero
            #user_traffic['GAT_LONG_LFEE']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_LONG_LFEE',aggfunc={'GAT_LONG_LFEE':'mean'})
            user_traffic['COUN_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'mean'})
            user_traffic['INTER_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'mean'})
            user_traffic['GAT_RAOM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            #Attention!!!lots values of FUNCATION_FEE are zeros only 20 not zero
            #user_traffic['FUNCATION_FEE']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_FEE',aggfunc={'FUNCATION_FEE':'mean'})

            #tr3 table
            user_traffic['FLUX_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'mean'})
            user_traffic['FREE_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'mean'})
            user_traffic['BILL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'mean'})
            user_traffic['LOCAL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'mean'})
            user_traffic['ROAM_CONT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'mean'})
            user_traffic['ROAM_INT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'mean'})
            user_traffic['TOTAL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'mean'})
            user_traffic['FREE_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'mean'})
            user_traffic['BILL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'mean'})
            user_traffic['TOTAL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'mean'})
            user_traffic['FREE_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'mean'})
            user_traffic['BILL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'mean'})

        else:
            #tr2 table窗口特征
            user_traffic['TOTAL_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'mean'})
            user_traffic['TOTAL_FEE_max'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'max'})
            user_traffic['TOTAL_FEE_min'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'min'})
            user_traffic['TOTAL_FEE_median'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'median'})

            user_traffic['BASE_RENT_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'mean'})
            user_traffic['BASE_RENT_FEE_max'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'max'})
            user_traffic['BASE_RENT_FEE_min'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'min'})
            user_traffic['BASE_RENT_FEE_median'] = data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'median'})

            user_traffic['DINNER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'mean'})
            user_traffic['DINNER_RENT_FEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'max'})
            user_traffic['DINNER_RENT_FEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'min'})
            user_traffic['DINNER_RENT_FEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'median'})

            user_traffic['FUNCATION_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'mean'})
            user_traffic['FUNCATION_RENT_FEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'max'})
            user_traffic['FUNCATION_RENT_FEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'min'})
            user_traffic['FUNCATION_RENT_FEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'median'})

            user_traffic['OTHER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'mean'})
            user_traffic['OTHER_RENT_FEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'max'})
            user_traffic['OTHER_RENT_FEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'min'})
            user_traffic['OTHER_RENT_FEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'median'})

            user_traffic['BASE_CALL_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'mean'})
            user_traffic['BASE_CALL_FEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'max'})
            user_traffic['BASE_CALL_FEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'min'})
            user_traffic['BASE_CALL_FEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'median'})

            user_traffic['COUN_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'mean'})
            user_traffic['COUN_LONG_LFEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'max'})
            user_traffic['COUN_LONG_LFEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'min'})
            user_traffic['COUN_LONG_LFEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'median'})

            user_traffic['INTER_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'mean'})
            user_traffic['INTER_LONG_LFEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'max'})
            user_traffic['INTER_LONG_LFEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'min'})
            user_traffic['INTER_LONG_LFEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'median'})

            #Attention!!!lots values of GAT_LONG_LFEE are zeros only 30 not zero
            #user_traffic['GAT_LONG_LFEE']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_LONG_LFEE',aggfunc={'GAT_LONG_LFEE':'mean'})
            user_traffic['COUN_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'mean'})
            user_traffic['COUN_ROAM_BFEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'max'})
            user_traffic['COUN_ROAM_BFEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'min'})
            user_traffic['COUN_ROAM_BFEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'median'})

            user_traffic['INTER_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'mean'})
            user_traffic['INTER_ROAM_BFEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'max'})
            user_traffic['INTER_ROAM_BFEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'min'})
            user_traffic['INTER_ROAM_BFEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'median'})

            user_traffic['GAT_RAOM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            user_traffic['GAT_RAOM_BFEE_max']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            user_traffic['GAT_RAOM_BFEE_min']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            user_traffic['GAT_RAOM_BFEE_median']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            #只有20个不为0
            #user_traffic['FUNCATION_FEE']=data['t2'].ix[data['t2']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FUNCATION_FEE',aggfunc={'FUNCATION_FEE':'mean'})

            #tr3 table窗口特征
            user_traffic['FLUX_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'mean'})
            user_traffic['FLUX_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'max'})
            user_traffic['FLUX_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'min'})
            user_traffic['FLUX_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'median'})

            user_traffic['FREE_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'mean'})
            user_traffic['FREE_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'max'})
            user_traffic['FREE_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'min'})
            user_traffic['FREE_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'median'})

            user_traffic['BILL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'mean'})
            user_traffic['BILL_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'max'})
            user_traffic['BILL_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'min'})
            user_traffic['BILL_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'median'})

            user_traffic['LOCAL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'mean'})
            user_traffic['LOCAL_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'max'})
            user_traffic['LOCAL_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'min'})
            user_traffic['LOCAL_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'median'})

            user_traffic['ROAM_CONT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'mean'})
            user_traffic['ROAM_CONT_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'max'})
            user_traffic['ROAM_CONT_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'min'})
            user_traffic['ROAM_CONT_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'median'})

            user_traffic['ROAM_INT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'mean'})
            user_traffic['ROAM_INT_NUM_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'max'})
            user_traffic['ROAM_INT_NUM_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'min'})
            user_traffic['ROAM_INT_NUM_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'median'})

            user_traffic['TOTAL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'mean'})
            user_traffic['TOTAL_FLUX_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'max'})
            user_traffic['TOTAL_FLUX_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'min'})
            user_traffic['TOTAL_FLUX_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'median'})

            user_traffic['FREE_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'mean'})
            user_traffic['FREE_FLUX_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'max'})
            user_traffic['FREE_FLUX_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'min'})
            user_traffic['FREE_FLUX_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'median'})

            user_traffic['BILL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'mean'})
            user_traffic['BILL_FLUX_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'max'})
            user_traffic['BILL_FLUX_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'min'})
            user_traffic['BILL_FLUX_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'median'})

            user_traffic['TOTAL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'mean'})
            user_traffic['TOTAL_DURA_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'max'})
            user_traffic['TOTAL_DURA_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'min'})
            user_traffic['TOTAL_DURA_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'median'})

            user_traffic['FREE_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'mean'})
            user_traffic['FREE_DURA_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'max'})
            user_traffic['FREE_DURA_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'min'})
            user_traffic['FREE_DURA_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'median'})

            user_traffic['BILL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'mean'})
            user_traffic['BILL_DURA_max']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'max'})
            user_traffic['BILL_DURA_min']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'min'})
            user_traffic['BILL_DURA_median']=data['t3'].ix[data['t3']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'median'})


        #tr5 table窗口特征
        user_traffic['DEPOSIT_MONEY_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'sum'})
        user_traffic['DEPOSIT_MONEY_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'mean'})
        user_traffic['DEPOSIT_MONEY_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'max'})
        user_traffic['DEPOSIT_MONEY_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'min'})
        user_traffic['DEPOSIT_MONEY_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'median'})
        user_traffic['tr5_count']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'count'})

        user_traffic['INIT_MONEY_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'sum'})
        user_traffic['INIT_MONEY_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'mean'})
        user_traffic['INIT_MONEY_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'max'})
        user_traffic['INIT_MONEY_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'min'})
        user_traffic['INIT_MONEY_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'median'})

        user_traffic['INVOICE_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'sum'})
        user_traffic['INVOICE_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'mean'})
        user_traffic['INVOICE_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'max'})
        user_traffic['INVOICE_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'min'})
        user_traffic['INVOICE_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'median'})

        user_traffic['PRINT_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'sum'})
        user_traffic['PRINT_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'mean'})
        user_traffic['PRINT_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'max'})
        user_traffic['PRINT_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'min'})
        user_traffic['PRINT_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'median'})

        user_traffic['OWE_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'sum'})
        user_traffic['OWE_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'mean'})
        user_traffic['OWE_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'max'})
        user_traffic['OWE_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'min'})
        user_traffic['OWE_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'median'})
        #处理奇数月偶数月金额
        user_traffic['oe_money_sum']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='oe_money',aggfunc={'oe_money':'sum'})
        user_traffic['oe_money_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='oe_money',aggfunc={'oe_money':'mean'})
        user_traffic['oe_money_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='oe_money',aggfunc={'oe_money':'max'})
        user_traffic['oe_money_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='oe_money',aggfunc={'oe_money':'min'})
        user_traffic['oe_money_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='oe_money',aggfunc={'oe_money':'median'})

        #销账日期间隔，包含default值
        user_traffic['cycle_gap_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'min'})
        user_traffic['cycle_gap_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'max'})
        user_traffic['cycle_gap_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'median'})
        user_traffic['cycle_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'mean'})
        #销账日期间隔，不包含default值
        user_traffic['cycle_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'min'})
        user_traffic['cycle_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'max'})
        user_traffic['cycle_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'median'})
        user_traffic['cycle_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'mean'})

        #cur_end_gap，包含default值
        user_traffic['cur_end_gap_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'min'})
        user_traffic['cur_end_gap_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'max'})
        user_traffic['cur_end_gap_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'median'})
        user_traffic['cur_end_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'mean'})
        #cur_end_gap，不包含default值
        user_traffic['cur_end_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'min'})
        user_traffic['cur_end_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'max'})
        user_traffic['cur_end_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'median'})
        user_traffic['cur_end_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'mean'})

        #cur_start_gap，包含default值
        user_traffic['cur_start_gap_min']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'min'})
        user_traffic['cur_start_gap_max']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'max'})
        user_traffic['cur_start_gap_median']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'median'})
        user_traffic['cur_start_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'mean'})
        #cur_start_gap，不包含default值
        user_traffic['cur_start_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'min'})
        user_traffic['cur_start_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'max'})
        user_traffic['cur_start_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'median'})
        user_traffic['cur_start_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID'].isin(time_gap_s))&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'mean'})




        #tr6 table窗口特征
        user_traffic['RECV_FEE_sum']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'sum'})
        user_traffic['RECV_FEE_mean']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'mean'})
        user_traffic['RECV_FEE_max']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'max'})
        user_traffic['RECV_FEE_min']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'min'})
        user_traffic['RECV_FEE_median']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'median'})
        user_traffic['tr6_count']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'count'})

        user_traffic['DAY_ID_MAX']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'max'})
        user_traffic['DAY_ID_MIN']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'min'})
        user_traffic['DAY_ID_MEAN']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'mean'})
        user_traffic['DAY_ID_MEDIAN']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'median'})

        user_traffic['CHANNEL_ID_num']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='CHANNEL_ID',aggfunc={'CHANNEL_ID':'nunique'})
        user_traffic['CHANNEL_ID_m']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].groupby('USER_ID').agg({'CHANNEL_ID':lambda x: x.value_counts().index[0]})
        user_traffic['PAYMENT_ID_num']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].pivot_table(index='USER_ID',values='PAYMENT_ID',aggfunc={'PAYMENT_ID':'nunique'})
        user_traffic['PAYMENT_ID_m']=data['t6'].ix[data['t6']['MONTH_ID'].isin(time_gap_s),:].groupby('USER_ID').agg({'PAYMENT_ID':lambda x: x.value_counts().index[0]})
        #窗口特征加前缀
        user_traffic = user_traffic.add_prefix(str(len(time_gap_s))+'mon_')



        #构造非窗口特征
        if time_gap in [201801,201712]:
            #tr2 table 非窗口特征
            user_traffic_nw['TOTAL_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='TOTAL_FEE',aggfunc={'TOTAL_FEE':'mean'})
            user_traffic_nw['BASE_RENT_FEE_mean'] = data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='BASE_RENT_FEE',aggfunc={'BASE_RENT_FEE':'mean'})
            user_traffic_nw['DINNER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DINNER_RENT_FEE',aggfunc={'DINNER_RENT_FEE':'mean'})
            user_traffic_nw['FUNCATION_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FUNCATION_RENT_FEE',aggfunc={'FUNCATION_RENT_FEE':'mean'})
            user_traffic_nw['OTHER_RENT_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OTHER_RENT_FEE',aggfunc={'OTHER_RENT_FEE':'mean'})
            user_traffic_nw['BASE_CALL_FEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='BASE_CALL_FEE',aggfunc={'BASE_CALL_FEE':'mean'})
            user_traffic_nw['COUN_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='COUN_LONG_LFEE',aggfunc={'COUN_LONG_LFEE':'mean'})
            user_traffic_nw['INTER_LONG_LFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INTER_LONG_LFEE',aggfunc={'INTER_LONG_LFEE':'mean'})
            #Attention!!!lots values of GAT_LONG_LFEE are zeros only 30 not zero
            #user_traffic_nw['GAT_LONG_LFEE']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='GAT_LONG_LFEE',aggfunc={'GAT_LONG_LFEE':'mean'})
            user_traffic_nw['COUN_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='COUN_ROAM_BFEE',aggfunc={'COUN_ROAM_BFEE':'mean'})
            user_traffic_nw['INTER_ROAM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INTER_ROAM_BFEE',aggfunc={'INTER_ROAM_BFEE':'mean'})
            user_traffic_nw['GAT_RAOM_BFEE_mean']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='GAT_RAOM_BFEE',aggfunc={'GAT_RAOM_BFEE':'mean'})
            #Attention!!!lots values of FUNCATION_FEE are zeros only 20 not zero
            #user_traffic_nw['FUNCATION_FEE']=data['t2'].ix[data['t2']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FUNCATION_FEE',aggfunc={'FUNCATION_FEE':'mean'})

            #tr3 table 非窗口特征
            user_traffic_nw['FLUX_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FLUX_NUM',aggfunc={'FLUX_NUM':'mean'})
            user_traffic_nw['FREE_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FREE_NUM',aggfunc={'FREE_NUM':'mean'})
            user_traffic_nw['BILL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='BILL_NUM',aggfunc={'BILL_NUM':'mean'})
            user_traffic_nw['LOCAL_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='LOCAL_NUM',aggfunc={'LOCAL_NUM':'mean'})
            user_traffic_nw['ROAM_CONT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='ROAM_CONT_NUM',aggfunc={'ROAM_CONT_NUM':'mean'})
            user_traffic_nw['ROAM_INT_NUM_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='ROAM_INT_NUM',aggfunc={'ROAM_INT_NUM':'mean'})
            user_traffic_nw['TOTAL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='TOTAL_FLUX',aggfunc={'TOTAL_FLUX':'mean'})
            user_traffic_nw['FREE_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FREE_FLUX',aggfunc={'FREE_FLUX':'mean'})
            user_traffic_nw['BILL_FLUX_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='BILL_FLUX',aggfunc={'BILL_FLUX':'mean'})
            user_traffic_nw['TOTAL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='TOTAL_DURA',aggfunc={'TOTAL_DURA':'mean'})
            user_traffic_nw['FREE_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='FREE_DURA',aggfunc={'FREE_DURA':'mean'})
            user_traffic_nw['BILL_DURA_mean']=data['t3'].ix[data['t3']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='BILL_DURA',aggfunc={'BILL_DURA':'mean'})


            #tr5 table非窗口特征
            user_traffic_nw['DEPOSIT_MONEY_sum']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'sum'})
            user_traffic_nw['DEPOSIT_MONEY_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'mean'})
            user_traffic_nw['DEPOSIT_MONEY_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'max'})
            user_traffic_nw['DEPOSIT_MONEY_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'min'})
            user_traffic_nw['DEPOSIT_MONEY_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'median'})
            user_traffic_nw['tr5_count']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DEPOSIT_MONEY',aggfunc={'DEPOSIT_MONEY':'count'})

            user_traffic_nw['INIT_MONEY_sum']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'sum'})
            user_traffic_nw['INIT_MONEY_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'mean'})
            user_traffic_nw['INIT_MONEY_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'max'})
            user_traffic_nw['INIT_MONEY_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'min'})
            user_traffic_nw['INIT_MONEY_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INIT_MONEY',aggfunc={'INIT_MONEY':'median'})

            user_traffic_nw['INVOICE_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'sum'})
            user_traffic_nw['INVOICE_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'mean'})
            user_traffic_nw['INVOICE_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'max'})
            user_traffic_nw['INVOICE_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'min'})
            user_traffic_nw['INVOICE_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='INVOICE_FEE',aggfunc={'INVOICE_FEE':'median'})

            user_traffic_nw['PRINT_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'sum'})
            user_traffic_nw['PRINT_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'mean'})
            user_traffic_nw['PRINT_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'max'})
            user_traffic_nw['PRINT_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'min'})
            user_traffic_nw['PRINT_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PRINT_FEE',aggfunc={'PRINT_FEE':'median'})

            user_traffic_nw['OWE_FEE_sum']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'sum'})
            user_traffic_nw['OWE_FEE_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'mean'})
            user_traffic_nw['OWE_FEE_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'max'})
            user_traffic_nw['OWE_FEE_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'min'})
            user_traffic_nw['OWE_FEE_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='OWE_FEE',aggfunc={'OWE_FEE':'median'})

            #销账日期间隔
            user_traffic_nw['cycle_gap_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'min'})
            user_traffic_nw['cycle_gap_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'max'})
            user_traffic_nw['cycle_gap_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'median'})
            user_traffic_nw['cycle_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'mean'})

            #销账日期间隔，不包含default值
            user_traffic_nw['cycle_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'min'})
            user_traffic_nw['cycle_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'max'})
            user_traffic_nw['cycle_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'median'})
            user_traffic_nw['cycle_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cycle_gap',aggfunc={'cycle_gap':'mean'})

            #cur_end_gap，包含default值
            user_traffic_nw['cur_end_gap_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'min'})
            user_traffic_nw['cur_end_gap_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'max'})
            user_traffic_nw['cur_end_gap_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'median'})
            user_traffic_nw['cur_end_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'mean'})
            #cur_end_gap，不包含default值
            user_traffic_nw['cur_end_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'min'})
            user_traffic_nw['cur_end_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'max'})
            user_traffic_nw['cur_end_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'median'})
            user_traffic_nw['cur_end_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_end_gap',aggfunc={'cur_end_gap':'mean'})

            #cur_start_gap，包含default值
            user_traffic_nw['cur_start_gap_min']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'min'})
            user_traffic_nw['cur_start_gap_max']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'max'})
            user_traffic_nw['cur_start_gap_median']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'median'})
            user_traffic_nw['cur_start_gap_mean']=data['t5'].ix[data['t5']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'mean'})
            #cur_start_gap，不包含default值
            user_traffic_nw['cur_start_gap_min_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'min'})
            user_traffic_nw['cur_start_gap_max_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'max'})
            user_traffic_nw['cur_start_gap_median_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'median'})
            user_traffic_nw['cur_start_gap_mean_nodefault']=data['t5'].ix[(data['t5']['MONTH_ID']==time_gap)&(data['t5']['START_CYCLE_ID']!=198001),:].pivot_table(index='USER_ID',values='cur_start_gap',aggfunc={'cur_start_gap':'mean'})



            #tr6 table非窗口特征
            user_traffic_nw['RECV_FEE_sum']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'sum'})
            user_traffic_nw['RECV_FEE_mean']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'mean'})
            user_traffic_nw['RECV_FEE_max']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'max'})
            user_traffic_nw['RECV_FEE_min']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'min'})
            user_traffic_nw['RECV_FEE_median']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'median'})
            user_traffic_nw['tr6_count']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='RECV_FEE',aggfunc={'RECV_FEE':'count'})

            user_traffic_nw['DAY_ID_MAX']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'max'})
            user_traffic_nw['DAY_ID_MIN']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'min'})
            user_traffic_nw['DAY_ID_MEAN']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'mean'})
            user_traffic_nw['DAY_ID_MEDIAN']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='DAY_ID',aggfunc={'DAY_ID':'median'})

            user_traffic_nw['CHANNEL_ID_num']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='CHANNEL_ID',aggfunc={'CHANNEL_ID':'nunique'})
            user_traffic_nw['CHANNEL_ID_m']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].groupby('USER_ID').agg({'CHANNEL_ID':lambda x: x.value_counts().index[0]})
            user_traffic_nw['PAYMENT_ID_num']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].pivot_table(index='USER_ID',values='PAYMENT_ID',aggfunc={'PAYMENT_ID':'nunique'})
            user_traffic_nw['PAYMENT_ID_m']=data['t6'].ix[data['t6']['MONTH_ID']==time_gap,:].groupby('USER_ID').agg({'PAYMENT_ID':lambda x: x.value_counts().index[0]})
            user_traffic_nw = user_traffic_nw.add_prefix(str(time_gap)+'_')

        user_traffic_all = pd.concat([user_traffic,user_traffic_nw],axis=1,sort=False)
        user_fea_concat.append(user_traffic_all)
    user_traffic_all = pd.concat(user_fea_concat,axis=1,sort=False)
    user_traffic_all = user_traffic_all.fillna(0)
    #merge with user_info
    user_traffic_all.reset_index(level='USER_ID')
    user_fea = user_info.merge(user_traffic_all,on='USER_ID',how='left')
    #transform to category
    cate_columns = ['CUST_TYPE','AREA_ID','CREDIT_CLASS','IS_ACCT','CERT_TYPE','CUST_SEX','age_bins','CONSTELLATION_DESC',
    '3mon_CHANNEL_ID_m','3mon_PAYMENT_ID_m','2mon_CHANNEL_ID_m','2mon_PAYMENT_ID_m','1mon_CHANNEL_ID_m','1mon_PAYMENT_ID_m',
    '201712_CHANNEL_ID_m','201712_PAYMENT_ID_m','201801_CHANNEL_ID_m','201801_PAYMENT_ID_m']
    j=0
    for col in user_fea['AREA_ID'].unique():
        user_fea.loc[user_fea['AREA_ID']==col,'AREA_ID']=j
        j = j+1
    j=0
    for col in user_fea['CONSTELLATION_DESC'].unique():
        user_fea.loc[user_fea['CONSTELLATION_DESC']==col,'CONSTELLATION_DESC']=j
        j = j+1
    le = LabelEncoder()
    for cate_col in cate_columns:
        user_fea.loc[user_fea[cate_col].isnull(),cate_col]=-1
        user_fea[cate_col]=le.fit_transform(user_fea[cate_col])
    if one_hot_on:
       cate_columns_oh = ['CUST_TYPE','AREA_ID','CREDIT_CLASS','CERT_TYPE','CONSTELLATION_DESC','3mon_CHANNEL_ID_m','3mon_PAYMENT_ID_m','2mon_CHANNEL_ID_m','2mon_PAYMENT_ID_m','1mon_CHANNEL_ID_m','1mon_PAYMENT_ID_m',
'201712_CHANNEL_ID_m','201712_PAYMENT_ID_m','201801_CHANNEL_ID_m','201801_PAYMENT_ID_m']
       user_fea = one_hot(user_fea,cate_columns_oh)
    user_fea_train = user_fea.loc[user_fea['IS_LOST'].isin([0,1]),:]
    user_fea_test = user_fea.loc[~user_fea['IS_LOST'].isin([0,1]),:]
    print ('user_fea_train length:',len(user_fea_train))
    print ('user_fea_test length:',len(user_fea_test))
    print ('feature generation is complete!')
    return user_fea_train,user_fea_test


def one_hot(user_fea,cate_columns_oh):
    user_fea[cate_columns_oh] = user_fea[cate_columns_oh].astype('str')
    dummy = pd.get_dummies(user_fea[cate_columns_oh])
    user_fea = user_fea.drop(cate_columns_oh,axis=1)
    user_fea = pd.concat([user_fea,dummy],axis=1,sort=False)
    return user_fea