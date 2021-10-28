import os
import numpy as np
import pandas as pd
from sklearn import model_selection

class HDFLoader():
    
    def __init__(self,datapath='./data.hdf5'):
        self.datapath = datapath
        with pd.HDFStore(self.datapath,mode='r') as store:
            self.train_batch_cnt = max([
                int(v.lstrip('/train').rstrip('data')) 
                for v in store.keys() 
                if v.startswith('/train') and v.endswith('data')
            ])+1
            self.val_batch_cnt = max([
                int(v.lstrip('/val').rstrip('data')) 
                for v in store.keys() 
                if v.startswith('/val') and v.endswith('data')
            ])+1
    
    def __len__(self,data_type='train'):
        if data_type not in {'train','val'}:
            raise Exception('data_type parameter must be either "train" or "val"')
        
        return self.train_batch_cnt if data_type=='train' else self.val_batch_cnt
    
    def get(self,data_type='train'):
        """
        Retrieve parsed HDF records.
        
        Parameters:
            data_type (str):     return data split from either 'train' or 'val'

        Returns:
            X (np.ndarray): np array of dataset features (returned as iterator item)
            y (np.ndarray): np array of dataset target(s) (returned as iterator item)
        
        """
        
        if data_type not in {'train','val'}:
            raise Exception('data_type parameter must be either "train" or "val"')
        
        batch_cnt = self.train_batch_cnt if data_type=='train' else self.val_batch_cnt
        
        with pd.HDFStore(self.datapath,mode='r') as store:
            for ix in range(batch_cnt):
                X = store.select(f'/{data_type}{ix}data')
                y = store.select(f'/{data_type}{ix}target').values.reshape(-1,1)

                yield X,y
                        
    def dump_data(self,data_type='train'):
        """
        Dumps all the data samples and their targets into two dataframes. 
        """
        
        if data_type not in {'train','val'}:
            raise Exception('datat_type parameter must be "train" or "val"')
        
        X_total = []
        y_total = []
        
        for (X_batch,y_batch) in self.get(data_type):
            X_total.append(X_batch)
            y_total.append(y_batch)
        
        X_total = np.concatenate(X_total,axis=0)
        y_total = np.concatenate(y_total,axis=0)
        
        return X_total,y_total
    
class HDFBuilder():
    
    def from_file(
        self,
        source_filepath='data/train.csv',
        output_filepath='data.hdf5',
        desired_batchsize = 50000,
        expected_workers = 10,
        test_sample_size = 0.15,
        shuffle_samples = True,
        stratify_data = True,
        features_to_exclude = ['id'],
        target_column_name = 'target'):

        # Don't build file if it already exists
        if os.path.exists(output_filepath):
            print(f'hdf5 file ({output_filepath}) already exists. No need to rebuild.')
            return

        # Load pandas Dataframe
        data = pd.read_csv(source_filepath)
        for entry in features_to_exclude:
            if entry in data:
                del data[entry]

        # Create train-test split
        stratified_colnames = data[target_column_name] if stratify_data else None
        data = model_selection.train_test_split(
            data,
            test_size=test_sample_size,
            shuffle=shuffle_samples,
            stratify=stratified_colnames,
        )

        # Split out target from features in training and test sets 
        X_train, X_val = data
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)

        y_train = X_train.pop(target_column_name)
        y_val = X_val.pop(target_column_name)
        train_sample_cnt, val_sample_cnt = X_train.shape[0], X_val.shape[0]

        # Publish minibatch chunks to hdf5 file
        with pd.HDFStore(output_filepath,'w') as f:

            microbatch_count = train_sample_cnt//desired_batchsize #*expected_workers
            for i in range(microbatch_count):
                tmp_x = X_train.iloc[i*desired_batchsize:(i+1)*desired_batchsize,:]
                tmp_y = y_train.iloc[i*desired_batchsize:(i+1)*desired_batchsize]
                f.append(f'train{str(i)}data', tmp_x)
                f.append(f'train{str(i)}target',tmp_y)

            microbatch_count = val_sample_cnt//desired_batchsize #*expected_workers
            for i in range(microbatch_count):
                tmp_x = X_val.iloc[i*desired_batchsize:(i+1)*desired_batchsize,:]
                tmp_y = y_val.iloc[i*desired_batchsize:(i+1)*desired_batchsize]
                f.append(f'val{str(i)}data', tmp_x)
                f.append(f'val{str(i)}target',tmp_y)

            
    def from_df(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        output_filepath='processed_data.hdf5',
        desired_batchsize = 50000,
        expected_workers = 10):

        # Don't build file if it already exists
        if os.path.exists(output_filepath):
            print(f'hdf5 file ({output_filepath}) already exists. No need to rebuild.')
            return

        train_sample_cnt, val_sample_cnt = X_train.shape[0], X_val.shape[0]

        # Publish minibatch chunks to hdf5 file
        with pd.HDFStore(output_filepath,'w') as f:

            microbatch_count = train_sample_cnt//desired_batchsize #*expected_workers
            for i in range(microbatch_count):
                tmp_x = X_train.iloc[i*desired_batchsize:(i+1)*desired_batchsize,:]
                tmp_y = y_train.iloc[i*desired_batchsize:(i+1)*desired_batchsize]
                f.append(f'train{str(i)}data', tmp_x)
                f.append(f'train{str(i)}target',tmp_y)

            microbatch_count = val_sample_cnt//desired_batchsize #*expected_workers
            for i in range(microbatch_count):
                tmp_x = X_val.iloc[i*desired_batchsize:(i+1)*desired_batchsize,:]
                tmp_y = y_val.iloc[i*desired_batchsize:(i+1)*desired_batchsize]
                f.append(f'val{str(i)}data', tmp_x)
                f.append(f'val{str(i)}target',tmp_y)