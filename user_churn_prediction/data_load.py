#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


def data_load(path):
    print ('training data is loading,hold on...')
    data={}
    tr1 = pd.read_table(path+'Contest2_Training_1.txt',delimiter='|',encoding='gbk',header=None)
    tr2 = pd.read_table(path+'Contest2_Training_2.txt',delimiter='|',encoding='gbk',header=None)
    tr3 = pd.read_table(path+'Contest2_Training_3.txt',delimiter='|',encoding='gbk',header=None)
    tr4 = pd.read_table(path+'Contest2_Training_4.txt',delimiter='|',encoding='utf-8',header=None)
    tr5 = pd.read_table(path+'Contest2_Training_5.txt',delimiter='|',encoding='gbk',header=None)
    tr6 = pd.read_table(path+'Contest2_Training_6.txt',delimiter='|',encoding='gbk',header=None)

    te1 = pd.read_table(path+'Contest2_Forecast_1.txt',delimiter='|',encoding='gbk',header=None)
    te2 = pd.read_table(path+'Contest2_Forecast_2.txt',delimiter='|',encoding='gbk',header=None)
    te3 = pd.read_table(path+'Contest2_Forecast_3.txt',delimiter='|',encoding='gbk',header=None)
    te4 = pd.read_table(path+'Contest2_Forecast_4.txt',delimiter='|',encoding='utf-8',header=None)
    te5 = pd.read_table(path+'Contest2_Forecast_5.txt',delimiter='|',encoding='gbk',header=None)
    te6 = pd.read_table(path+'Contest2_Forecast_6.txt',delimiter='|',encoding='gbk',header=None)

    column_1 = ['MONTH_ID','USER_ID','CUST_ID','CUST_TYPE','SERVICE_TYPE','BRAND_ID','AREA_ID',
    'USER_DIFF_CODE','DEVICE_NUMBER','NET_TYPE_CBSS','SCORE_VALUE','CREDIT_CLASS','BASIC_CREDIT_VALUE','CREDIT_VALUE',
    'IS_ACCT','PAY_MODE','INNET_DATE','OPER_DATE','OPEN_MODE','OPEN_DEPART_ID','IN_DEPART_ID','IS_LOST']

    column_2 = ['MONTH_ID','PROV_ID','USER_ID','CUST_ID','TOTAL_FEE','BASE_RENT_FEE','DINNER_RENT_FEE',
    'FUNCATION_RENT_FEE','OTHER_RENT_FEE','BASE_CALL_FEE','COUN_LONG_LFEE','INTER_LONG_LFEE','GAT_LONG_LFEE','COUN_ROAM_BFEE',
    'INTER_ROAM_BFEE','GAT_RAOM_BFEE','FUNCATION_FEE','WIRLESS_CARD_FEE','WLAN_FEE',
    'CHARGE_FAV_FEE','ADJUST_FEE','DISCOUNT_FEE','GRANT_FEE']
    column_3 = ['MONTH_ID','USER_ID','FLUX_NUM','FREE_NUM','BILL_NUM','LOCAL_NUM','ROAM_PROV_NUM','ROAM_CONT_NUM',
    'ROAM_GAT_NUM','ROAM_INT_NUM','TOTAL_FLUX','FREE_FLUX','BILL_FLUX','TOTAL_DURA','FREE_DURA','BILL_DURA','TOTAL_FEE_DATA'
    ,'USER_ID_OLD']
    column_4 = ['CUST_ID','CERT_TYPE','CUST_SEX','CUST_BIRTHDAY','CONSTELLATION_DESC','CERT_AGE','MONTH_ID']
    column_5 = ['MONTH_ID','USER_ID','DEPOSIT_MONEY','INIT_MONEY','ODD_MONEY','EVEN_MONEY','INVOICE_FEE',
    'PRINT_FEE','START_CYCLE_ID','END_CYCLE_ID','OWE_FEE','VALID_FLAG','FREEZE_FEE']
    column_6 = ['MONTH_ID','DAY_ID','CUST_ID','USER_ID','CHANNEL_ID','PAYMENT_ID','RECV_FEE']

    tr1.columns,te1.columns=column_1,column_1
    tr2.columns,te2.columns=column_2,column_2
    tr3.columns,te3.columns=column_3,column_3
    tr4.columns,te4.columns=column_4,column_4
    tr5.columns,te5.columns=column_5,column_5
    tr6.columns,te6.columns=column_6,column_6

    data['t1'] = pd.concat([tr1,te1],axis=0,sort=False)
    data['t2'] = pd.concat([tr2,te2],axis=0,sort=False)
    data['t3'] = pd.concat([tr3,te3],axis=0,sort=False)
    data['t4'] = pd.concat([tr4,te4],axis=0,sort=False)
    data['t5'] = pd.concat([tr5,te5],axis=0,sort=False)
    data['t6'] = pd.concat([tr6,te6],axis=0,sort=False)

    drop_col = {}
    drop_col['t1'] = ['MONTH_ID','SERVICE_TYPE','BRAND_ID','USER_DIFF_CODE','DEVICE_NUMBER','NET_TYPE_CBSS','BASIC_CREDIT_VALUE','CREDIT_VALUE','PAY_MODE'
    ,'OPEN_MODE','OPEN_DEPART_ID','IN_DEPART_ID']
    drop_col['t2'] = ['PROV_ID','GAT_LONG_LFEE','FUNCATION_FEE','WIRLESS_CARD_FEE','WLAN_FEE','CHARGE_FAV_FEE','ADJUST_FEE','DISCOUNT_FEE','GRANT_FEE']
    drop_col['t3'] = ['ROAM_PROV_NUM','ROAM_GAT_NUM','USER_ID_OLD']
    drop_col['t4'] = ['CUST_BIRTHDAY','MONTH_ID']
    drop_col['t5']= ['VALID_FLAG','FREEZE_FEE']
    drop_col['t6'] = []
    #drop columns and label category features
    for key in drop_col.keys():
        data[key] = data[key].drop(drop_col[key],axis=1)
    print ('Training data load complete!')
    return data