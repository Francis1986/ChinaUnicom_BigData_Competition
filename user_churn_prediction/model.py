
#!/usr/bin/python
# -*- coding:utf-8 -*-

import lightgbm as lgb
import xgboost as xgb



def lgb_predict(x_train,y_train,x_test,param):
    num_boost_round = 2000000000
    early_stopping_rounds = 10
    lgb_train = lgb.Dataset(x_train,y_train)
    lgb_gbm=lgb.train(param,lgb_train,valid_sets=lgb_train,num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds)
    y_pred = lgb_gbm.predict(x_test,num_iteration=lgb_gbm.best_iteration)
    return y_pred,lgb_gbm



def xgb_predict(x_train,y_train,x_test,y_test,param):
    num_boost_round = 2000000000
    early_stopping_rounds = 10
    xgb_train  = xgb.DMatrix(x_train.values,label=y_train.values)
    xgb_test = xgb.DMatrix(x_test.values,label=y_test.values)
    watch_list = [(xgb_test,'evals'), (xgb_train,'train')]
    xgb_gbm = xgb.train(param,xgb_train,evals=watch_list,num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds)     
    y_pred = xgb_gbm.predict(xgb_test)
    return y_pred,xgb_gbm