import gc
import os
import pickle

import hyperopt
from hyperopt import fmin, tpe
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import numpy as np
from sklearn import metrics,model_selection

from models import LGBModel,XGBModel
from hdf_utils import HDFDatasetBuilder,HDFLoader


class ModelBuilder():
    """
    Trains an LGBM or XGBM model using hyperopt library and publishes the model to an MLFlow library.
    This library also includes a helper functions for selecting the best n hyperparameter configs.
    """
    
    MLFLOW_URI='postgresql://mlflow:mlflow@localhost:5432/mlflow'
    ARTIFACT_STORE='/media/ryan/aloha/mlflow'
    SOURCE_DATA='./data.hdf5'
    
    
    def __init__(self): 
        pass
    
    
    def init_mlflow_run(self):
        """
        Using the argparse inputs, this function will initialize the mlflow experiment and run objects.
        If necessary, this function also creates a new experiment object.

        Output: (experiment_id, run_id)
        """

        mlflow.set_tracking_uri(self.MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Create experiment if it doesn't already exist
        if not client.get_experiment_by_name(self.experiment_name):
            mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(self.experiment_name)
        
        experiment_id = (
            client.get_experiment_by_name(self.experiment_name)
            .experiment_id
        )

        experiment_run = client.create_run(experiment_id)
        run_id = experiment_run.info.run_id

        return experiment_id, run_id

    
    def _run_training_cycle(self, args):
        """
        Runs a complete training cycle for provided model_type and hyperparameter args.
        This method does the following steps:
            1) initialize model
            2) Setup mlflow instrumentation / hooks
            3) Create cross-validation splits & trains model on n cv-splits
            4) Computes the average metric score (auc)
            5) Retrains the model on the full dataset 
            6) Stores everything to mlflow
            
        Params:
            args (dict): dictionary of all the hyperparameters to be used for this model type
        """
        
        if self.model_type.lower()=='xgbmodel':
            model = XGBModel(args)
        elif self.model_type.lower()=='lgbmodel':
            model = LGBModel(args)
        else:
            raise Exception('model_type must be either "XGBModel" or "LGBModel"')
        
        print(model)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_DATABASE_URI"))
        experiment_id,run_id = self.init_mlflow_run()
        
        # Assign storage directory
        artifact_dir = os.path.join(
            self.ARTIFACT_STORE,
            experiment_id,
            run_id,
            'artifacts'
        )
        
        with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
        
            # Log custom parameters in mlflow dashboard
            mlflow.set_tag('mlflow.user','andersonvc')
            for k, v in model.params.items():
                mlflow.log_param(k, v)
        
            # load data from disk
            X,y = HDFLoader().dump_data('train')

            # Generate stratified split
            auc_scores = []
            skfold = model_selection.StratifiedShuffleSplit(n_splits=2,test_size=0.2)
            for train_ix,val_ix in skfold.split(X,y):
                
                X_train,y_train = X[train_ix],y[train_ix].ravel()
                X_val,y_val = X[val_ix], y[val_ix].ravel()
                
                model.fit(X_train,y_train)

                y_pred = model.predict(X_val)
                auc_score = metrics.roc_auc_score(y_val,y_pred)
                
                auc_scores.append(auc_score)

            combined_auc_score = np.mean(auc_scores)
            print('combined_score',combined_auc_score)
            
            # Retrain model on the full train/val dataset & store model
            model.fit(X_train,y_train)
            mlflow.log_metric(key="auc", value=combined_auc_score)
            model.save(artifact_dir)
            
        del X_train,y_train,X_val,y_val,X,y
        gc.collect()
        
        return -1.0 * combined_auc_score
    
    
    def get_best_models(self, experiment_name, model_type, top_n=1):
        """
        Returns a list of dictionaries for top_n models
        
        params:
            experiment_name (str): MLFlow experiment directory to search
            top_n (int): returns N best results
        
        returns: 
            list(dict) -> {'score':<auc_score>, 'path':<model_path>}
        """
        
        if model_type=='xgbmodel':
            model_name = 'model.xgb'
        elif model_type=='lgbmodel':
            model_name = 'model.lgb'
        else:
            raise Exception('model_type must be either "xgbmodel" or "lgbmodel"')

        mlflow.set_tracking_uri(self.MLFLOW_URI)
        client = mlflow.tracking.client.MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        
        results = client.search_runs(
          experiment_ids=experiment_id,
          filter_string="",
          run_view_type=ViewType.ACTIVE_ONLY,
          max_results=top_n,
          order_by=["metrics.auc DESC"]
        )
        
        results = [(v.data.metrics['auc'],v.info.run_id) for v in results if 'auc' in v.data.metrics]
        
        path_builder = lambda run_id: os.path.join(self.ARTIFACT_STORE,experiment_id,run_id,'artifacts',model_name)

        return [{'score':score,'path':path_builder(run_id)} for score,run_id in results]
    
    
    def run(self,experiment_name, model_type, hyperopt_space):
        """
        Loads a local hyperopt record & runs a tree of parzen estimator (TPE) to get the best hyperparam trial config.
        Results are published to the MLFlow server.
        
        params: 
            experiment_name (str): MLFlow experiment directory where results are stored
            model_type (str): This is either 'XGBModel' or 'LGBModel'
            hyperopt_space (dict): dict of all configurable hyperopt space ranges
        
        """

        self.experiment_name = experiment_name
        self.model_type = model_type
        
        hyperopt_filename = f"{self.experiment_name}.hyperopt"

        if os.path.isfile(hyperopt_filename):
            trials = pickle.load(open(hyperopt_filename,'rb'))
            trial_cnt = len(trials.trials)+1
        else:
            trials = hyperopt.Trials()
            trial_cnt = 1
        
        # Run the experiment loop
        fmin(
            fn=self._run_training_cycle, 
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=trial_cnt,
            trials=trials,
        )

        # save the trials object
        with open(hyperopt_filename, "wb") as f:
            pickle.dump(trials, f)