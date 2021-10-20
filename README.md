# Dead Simple Boosting Ensembler
###### (Using XGBoost, LightGBM, pytorch, sklearn, mlflow, hyperopt, & hdf5)
The 'boost-ensembler' notebook was prepared for the Kaggle October 2021 tabular-data competition, with the objective of creating a high performing model with minimal manual effort. Therefore, we're going with an ensemble of boosting models. As the tree construction for XGBoost and LightGBM are markedly different, it makes sense to ensemble the best tuned models from each of these libraries. Additionally, it might also be useful to include a tablular-NN model to our ensemble (if time permits). The competition data is pretty simple, it's a binary classification problem with a mixture of normalized float & boolean features. The data is balanced & scored via ROC auc. 

Behind the scenes, this script does some pretty cool stuff --at least I think so

1) Converts source data to a HDF5 table. The hdf5 data is grouped like '/{train|test}{0-N}{data|target}', The {0-N} seems a little wonky, but it allows the data to be further separated into minibatches, which makes multiprocessing super easy. Combining this with pytorch's Dataloader cuts the time to load our data into memory from 60s (using the original csv file) to 3s. The {train|test} split is automatically stratified (even though it's not necessary for this dataset).

2) The ModelBuilder has a hook for our MLFlow server. This makes keeping track of experiments much easier. After running this script for a while, we can run 'get_best_models()' which will return the N best performing models. This is especially useful later on when we want to build out our ensemble.

3) ModelBuilder also incorporates hyperopt. You'll still need to define the models' valid hyperparameter search space, but we now get to use TPE when determining our next search params, a massive improvement over grid-search or random search.

4) The models (XGBModel or LGBModel) are trained using the GPU. We use kfold cross-validation on our training data (split in to train/val) to compute the run's average auc metric. Once finished the model is retrained using the entire training data & uploaded to the MLFlow artifact store.

5) Finally, we can query the MLFlow server to get the top N XGBModels and top N LGBModels. Assign weights to them and plug them into our ensembler. (Not coded out yet, but the logic is pretty simple).

6) Done.

### Getting started
1) The dependencies in this project were versioned using poetry. To launch the notebook, you'll only need to run:
```poetry install && poetry run jupyer lab```
2) The script expects a running MLFlow server. By default, we're running a docker image for our MLFlow server, which can be accessed at ```postgresql://mlflow:mlflow@localhost:5432/mlflow``` and its corresponding artifact store is a directory on the host machine: ```/media/ryan/aloha/mlflow```
3) The raw data (ie the comptetion csv file) should be accessible from this directory. I've added a local symlink directory called data, which mounts all the comptetion data.
4) Both boost models use the GPU. They can still be run using CPU, but it'll be painfully slow.