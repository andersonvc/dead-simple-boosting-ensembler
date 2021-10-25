import os
import numpy as np

from models import XGBModel,LGBModel

def build_ensemble(model_list,X,y,ensembler_dir='./final_models'):
    
    if os.path.isdir(ensembler_dir):
        if os.listdir(ensembler_dir):
            raise Exception(f'{ensembler_dir} directory needs to be deleted or empty to retrain ensembled models.')
    else:
        os.mkdir(ensembler_dir)
    
    
    weight_values = np.array([model['score'] for model in model_list])
    np.savetxt(f'{ensembler_dir}/weights.csv', weight_values, delimiter=',')
    
    for i,model_data in enumerate(model_list):
        filepath = model_data['path']
        filetype = filepath.split('.')[-1]
        params = model_data['params']

        output_filepath = f'{ensembler_dir}/{i}'

        if filetype=='xgb':
            model=XGBModel()
            model.load(filepath,params)
        elif filetype=='lgb':
            model=LGBModel()
            model.load(filepath,params)
        else:
            raise Exception('stored model type must be .xgb or .lgb')
        
        os.mkdir(f'{ensembler_dir}/{i}')

        model.fit(X,y)
        model.save(output_filepath,use_mlflow=False)
        del model
        gc.collect()


def ensemble_predict(X,ensembler_dir='./final_models'):
    
    res = []
    
    model_weights = np.loadtxt(f'{ensembler_dir}/weights.csv')
    model_weights = model_weights/sum(model_weights)

    for i,weight in enumerate(model_weights):
        
        model_dir = f'{ensembler_dir}/{i}'

        if os.path.exists(f'{model_dir}/model.xgb'):
            model=XGBModel()
            model.load(model_dir)
            y_pred = model.predict(X)
        else:
            model=LGBModel()
            model.load(model_dir)
            y_pred = model.predict(X)

        res.append(y_pred*weight)
    
    return sum(res)