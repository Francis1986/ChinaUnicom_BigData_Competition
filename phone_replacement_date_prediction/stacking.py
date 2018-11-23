import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,recall_score,precision_score,mean_squared_error,median_absolute_error
from model import train_lgb,train_xgb,train_LogReg,train_LinReg
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from glob import glob

class stacking_ensemble(object):
    def __init__(self,cv=5,random_state=0,model_list_layer1=['lgb','xgb'],task='clf',score='auc'):
        self.cv = cv
        self.model_list_layer1 = model_list_layer1
        #定义模型结构体
        self.model_struct = {}
        #如果是分类任务第二层模型默认使用逻辑回归
        self.task = task
        self.random_state = random_state
        self.score = score

    def fit(self,X,y):
        #输入要求为pandas_dataframe格式的数据
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        #第一层训练,xgb,lgb
        print ('开始stacking第一层训练，请稍后... ...')
        self.model_struct['layer1']={}
        y_pred_layer1 = pd.DataFrame()
        y_pred_layer1['label'] = y
        metric_list_lgb = []
        metric_list_xgb = []
        for model in self.model_list_layer1:
            print ('stacking第一层{}模型开始训练，请稍后... ...'.format(model))
            self.model_struct['layer1'][model]=[]
            kf = KFold(n_splits=self.cv,shuffle=True,random_state=self.random_state)
            fmp_list = []
            y_pred_layer1['y_pred_{}'.format(model)] = 0
            i = 0
            for train_index,test_index in kf.split(np.arange(0,len(X))):
                x_train = X.loc[train_index]
                y_train = y.loc[train_index]
                x_test = X.loc[test_index]
                y_test = y.loc[test_index]
                if model =='lgb':
                    model_lgb = train_lgb(x_train,y_train,x_test,y_test)
                    self.model_struct['layer1'][model].append(model_lgb)
                    y_pred = model_lgb.predict(x_test,num_iteration = model_lgb.best_iteration)
                    y_pred_layer1.loc[test_index,'y_pred_{}'.format(model)]=y_pred
                    #计算测量误差
                    if self.score == 'auc':
                        score = roc_auc_score(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    elif self.score == 'f1-score':
                        score = f1_score(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    elif self.score == 'accuracy':
                        score = accuracy_score(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    elif self.score == 'recall':
                        score= recall_score(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    elif self.score == 'mse':
                        score = mean_squared_error(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    elif self.score == 'mae':
                        score = median_absolute_error(y_test.values,y_pred)
                        metric_list_lgb.append(score)
                    #计算KS统计量
                    # ks = KS(y_test.values,y_pred)
                    print ('{}分值为:{}'.format(self.score,score))
                    # print ('KS_value:{}'.format(ks))
                    #lgb特征重要性保存
                    fmp =  pd.DataFrame(index=X.columns)
                    fmp['imp_{}_{}'.format(model,i)]=model_lgb.feature_importance()
                    fmp.sort_values('imp_{}_{}'.format(model,i),ascending=False)
                    fmp_list.append(fmp)
                if model == 'xgb':
                    model_xgb = train_xgb(x_train,y_train,x_test,y_test)
                    self.model_struct['layer1'][model].append(model_xgb)
                    y_pred = model_xgb.predict(xgb.DMatrix(x_test.values))
                    y_pred_layer1.loc[test_index,'y_pred_{}'.format(model)]=y_pred
                    model_xgb.feature_names = list(X.columns)
                    #计算测量误差
                    if self.score == 'auc':
                        score = roc_auc_score(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    elif self.score == 'f1-score':
                        score = f1_score(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    elif self.score == 'accuracy':
                        score = accuracy_score(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    elif self.score == 'recall':
                        score= recall_score(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    elif self.score == 'mse':
                        score = mean_squared_error(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    elif self.score == 'mae':
                        score = median_absolute_error(y_test.values,y_pred)
                        metric_list_xgb.append(score)
                    #计算KS统计量
                    # ks = KS(y_test.values,y_pred)
                    print ('{}分值为:{}'.format(self.score,score))
                    # print ('KS_value:{}'.format(ks))
                    fmp =  pd.DataFrame({'fmp_{}_{}'.format(model,i):list(model_xgb.get_score().values())},index = list(model_xgb.get_score().keys()))
                    fmp_list.append(fmp)
                i+=1
            #第一层交叉验证特征打分保存
            if model in ['lgb','xgb']:
                fmp = pd.concat(fmp_list,axis=1,sort=False)
                fmp = fmp.fillna(0)
                fmp['mean_score_{}'.format(model)] = fmp.mean(axis=1)
                fmp = fmp.sort_values('mean_score_{}'.format(model),ascending = False)
                fmp.to_csv('./fmp/fmp_{}.csv'.format(model))
            print ('stacking第一层{}模型训练完毕！'.format(model))
        print ('lgb模型{}评分:{}'.format(self.score,metric_list_lgb))
        print ('xgb模型{}评分:{}'.format(self.score,metric_list_xgb))
        y_pred_layer1.to_csv('y_layer1.csv')
        print ('stacking第一层所有模型训练完毕！')
        #第二层训练，分类使用LR作为第二层模型
        print ('stacking第二层训练开始，请稍后... ...')
        if self.task == 'clf': #暂时只支持分类任务
            self.model_struct['layer2']=[]
            x_train_layer2 = y_pred_layer1.drop('label',axis = 1)
            y_train_layer2 = y_pred_layer1['label']
            model_layer2_logreg = train_LogReg(x_train_layer2,y_train_layer2)
            self.model_struct['layer2'].append(model_layer2_logreg)
        if self.task == 'reg': #暂时只支持分类任务
            self.model_struct['layer2']=[]
            x_train_layer2 = y_pred_layer1.drop('label',axis = 1)
            y_train_layer2 = y_pred_layer1['label']
            model_layer2_linreg = train_LinReg(x_train_layer2,y_train_layer2)
            self.model_struct['layer2'].append(model_layer2_linreg)
        print ('stacking第二层训练完毕！')

    def predict(self,X_test):
        y_pred_df = pd.DataFrame()
        #第一层预测
        for model_name in self.model_list_layer1:
            column_layer1 = []
            for i,model_sub in enumerate(self.model_struct['layer1'][model_name]):
                if model_name == 'lgb':
                   y_pred_layer1_lgb = model_sub.predict(X_test,num_iteration=model_sub.best_iteration)
                   y_pred_df['lgb_{}'.format(i)] = y_pred_layer1_lgb
                   column_layer1 .append('lgb_{}'.format(i))
                if model_name == 'xgb':
                   y_pred_layer1_xgb = model_sub.predict(xgb.DMatrix(X_test))
                   y_pred_df['xgb_{}'.format(i)] = y_pred_layer1_xgb
                   column_layer1 .append('xgb_{}'.format(i))
            print ("column_layer1:",column_layer1)
            y_pred_df[model_name +'_mean'] = y_pred_df[column_layer1].mean(axis=1)

        #第二层预测
        column_layer2 =[]
        for col in y_pred_df.columns:
            if 'mean' in col:
                column_layer2.append(col)
        # print (y_pred_df)
        # print ("column_layer2:",column_layer2)
        y_pred_layer2 = y_pred_df[column_layer2]
        if self.task == 'clf':
            y_pred_final = self.model_struct['layer2'][0].predict_proba(y_pred_layer2.values)
            return y_pred_final[:,1]
        if self.task == 'reg':
            y_pred_final = self.model_struct['layer2'][0].predict(y_pred_layer2.values)
            return y_pred_final



    def save_model(self,file_path='./model_stacking/'):
        print ('stacking模型保存中，请稍后... ...')
        #保存第一层模型
        print ('stacking第一层模型保存中... ...')
        for key in self.model_struct['layer1'].keys():
            for i,model in enumerate(self.model_struct['layer1'][key]):
                model.save_model('{}model_layer1_{}_{}.model'.format(file_path,key,i))
        print ('stacking第一层模型保存完毕！')
        #保存第二层模型，暂时只支持分类任务
        print ('stacking第二层模型保存中... ...')
        for model in self.model_struct['layer2']:
            if self.task == 'clf':
                joblib.dump(model,'{}model_layer2_logreg.pkl'.format(file_path))
            if self.task == 'reg':
                joblib.dump(model,'{}model_layer2_linreg.pkl'.format(file_path))
        print ('stacking第二层模型保存完毕！')
        print ('stacking模型保存完毕')

    def load_model(self,file_path='./model_stacking/'):
        #读取第一层模型文件名
        model_file_list_layer1 = glob('{}*layer1*'.format(file_path))
        #读取第二层模型文件名
        model_file_list_layer2 = glob('{}*layer2*'.format(file_path))
        self.model_struct = {}
        #导入第一层模型
        self.model_struct['layer1'] = {}
        for model in self.model_list_layer1:
             self.model_struct['layer1'][model] = []
        for model_layer1_sub in model_file_list_layer1:
            if 'lgb' in model_layer1_sub:
                model_lgb = lgb.Booster(model_file=model_layer1_sub)
                self.model_struct['layer1']['lgb'].append(model_lgb)
            if 'xgb' in model_layer1_sub:
                model_xgb = xgb.Booster(model_file=model_layer1_sub)
                self.model_struct['layer1']['xgb'].append(model_xgb)
        #导入第二层模型
        self.model_struct['layer2'] = []
        if self.task == 'clf':
            for model_layer2_sub in model_file_list_layer2:
                if 'logreg' in model_layer2_sub:
                    model_layer2 = joblib.load(model_layer2_sub)
                    self.model_struct['layer2'].append(model_layer2)
        if self.task == 'reg':
            for model_layer2_sub in model_file_list_layer2:
                if 'linreg' in model_layer2_sub:
                    model_layer2 = joblib.load(model_layer2_sub)
                    self.model_struct['layer2'].append(model_layer2)
        return self.model_struct