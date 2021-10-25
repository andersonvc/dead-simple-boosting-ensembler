import datetime
import mlflow
import lightgbm
import xgboost as xgb
import os

def convert_str_to_num(var:str):
    if var.isnumeric():
        return int(var)    
    try:
        return float(var)
    except:
        return var


class Model(object):
    def __init__(self,args):
        pass
    
    def save(self,directory,use_mlflow):
        raise NotImplementedError
    
    def load(self,directory):
        raise NotImplementedError
    
    def predict(self,X):
        raise NotImplementedError
    
    def fit(self,X,y):
        raise NotImplementedError
        

class LGBModel(object):
    def __init__(self,args={}):
        super().__init__()
        
        self.model_name = 'model.lgb'
        
        default_params = {
            'num_leaves':31,
            'max_depth':-1,
            'learning_rate':0.1,
            'n_estimators':1000,
            'objective':'binary',
            'reg_alpha':0.,
            'reg_lambda':0.,
            'subsample':1.,
            'device':'gpu',
            'random_state':datetime.datetime.now().microsecond,
        }
        
        self.params = {**default_params,**args}
        print(self.params)
        self.model = lightgbm.LGBMRegressor(**self.params)
                
    def save(self,directory,use_mlflow=True):
        fullpath = os.path.join(directory,self.model_name)
        if use_mlflow:
            mlflow.lightgbm.save_model(self.model.booster_, fullpath)
        else:
            self.model.booster_.save_model(fullpath)

    def load(self,directory,params=None):
        fullpath = os.path.join(directory, self.model_name)
        self.model = lightgbm.Booster(model_file=fullpath)
        if params:
            self.params = {k:convert_str_to_num(v) for k,v in params.items()}

    def predict(self,X):
        return self.model.predict(X)
    
    def fit(self,X,y):
        self.params['random_state'] = datetime.datetime.now().microsecond
        self.model = lightgbm.LGBMRegressor(**self.params)
        print(self.params)
        self.model.fit(X,y,eval_metric=['auc'])

        
class XGBModel(object):
    def __init__(self,args={}):
        super().__init__()
        
        self.model_name = 'model.xgb'
        
        default_params = {
            'n_estimators':1000,
            'max_depth':6,
            'tree_method':'gpu_hist',
            'use_label_encoder':False,
            'subsample':0.5,
            'objective':'binary:logistic',
            'seed':datetime.datetime.now().microsecond,
        }
        
        self.params = {**default_params,**args}
        self.model = xgb.XGBRegressor(**self.params)

    def save(self,directory,use_mlflow=True):
        fullpath = os.path.join(directory,self.model_name)
        if use_mlflow:
            mlflow.xgboost.save_model(self.model, fullpath)
        else:
            self.model.save_model(fullpath)

    def load(self,directory,params=None):
        fullpath = os.path.join(directory, self.model_name)
        self.model.load_model(fullpath)
        if params:
            self.params = {k:convert_str_to_num(v) for k,v in params.items()}

    def predict(self,X):
        return self.model.predict(X)
    
    def fit(self,X,y):
        self.params['seed'] = datetime.datetime.now().microsecond
        self.model = xgb.XGBRegressor(**self.params)
        print(self.params)
        self.model.fit(X,y,eval_metric=['auc'])
