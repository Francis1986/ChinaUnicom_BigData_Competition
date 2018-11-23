import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet



def train_xgb(x_train,y_train,x_test,y_test):
    booster_param = {
        'objective':'reg:linear', #用于二分类任务
        'eval_metric':'rmse', #损失函数使用对数损失，这个可以自定义
        'eta':0.01, #学习率
        'gamma':0, #最小可分裂增益，只有增益达到gamma值时，该节点才进行分类，用于控制过拟合
        'max_depth':8, #树的最大深度
        'min_child_weight':0.001, #节点最小可分裂的hession值，同样用于处理过拟合
        'colsample_bytree':1, #列采样比例
        'subsample':1, #行采样比例
        'lambda':1, #l2正则系数
        'alpha':0, #l1正则系数
        'silent':1
        }
    xgb_train  = xgb.DMatrix(x_train.values,label=y_train.values)
    xgb_test = xgb.DMatrix(x_test.values,label=y_test.values)
    watch_list = [(xgb_train,'train'),(xgb_test,'evals')]
    param_general = dict(
        evals=watch_list,
        verbose_eval=False,
        num_boost_round=200000,
        early_stopping_rounds=10
        )
    model_xgb = xgb.train(params=booster_param,dtrain=xgb_train,**param_general)
    return model_xgb

def train_lgb(x_train,y_train,x_test,y_test):
    booster_param = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'learning_rate': 0.01,
        'num_leaves':40,
        'min_data_in_leaf':20,
        'min_sum_hessian_in_leaf':0.001,#最小可分裂的hession值
        'bagging_fraction': 1,
        'feature_fraction':0.9,
        'bagging_freq':5,#只有在bagging fraction不为1的时候起作用
        'lambda_l1':0,
        'lambda_l2':0
        }
    lgb_train  = lgb.Dataset(x_train,label=y_train) #lgb支持直接读取pandas数据
    lgb_test = lgb.Dataset(x_test,label=y_test)
    watch_list = [lgb_test,lgb_train]
    param_general = dict(
        valid_sets=watch_list, #观察的数据集，放到最后一个为观察对象，对象损失小于一定阈值达到一定步数时停止驯良
        verbose_eval=False,  #是否打印损失值
        num_boost_round=200000, #迭代次数
        early_stopping_rounds =10  #早停止次数
        )
    model_lgb = lgb.train(params=booster_param,train_set=lgb_train,**param_general)
    return model_lgb

def train_LogReg(x_train,y_train):
    param = {
        'penalty':'l2',#正则化方式
        'dual':False,
        #Dual or primal formulation.
        #Dual formulation is only implemented for l2 penalty with liblinear solver.
        #Prefer dual=False when n_samples > n_features.
        'tol':1e-4,
        #Tolerance for stopping criteria
        'C':1,#正则化参数
        'fit_intercept':True,
        'intercept_scaling':1,
        'class_weight':None,
        # Weights associated with classes in the form {class_label: weight}.
        # If not given, all classes are supposed to have weight one.
        # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to
        # class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        # Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        # New in version 0.17: class_weight=’balanced’
        'random_state':None,
        'solver':'liblinear',
        'max_iter':100,
        'multi_class':'ovr',
        'verbose':1,
        'n_jobs':None
    }
    model_logreg = LogReg(**param).fit(x_train,y_train)
    return model_logreg


def train_LinReg(x_train,y_train):
    param = {
      'alpha':2,
      'fit_intercept':True,
      'normalize':False,
      'copy_X':True
      #'n_jobs':None
    }
    model_linreg = Lasso(**param).fit(x_train,y_train)
    return model_linreg