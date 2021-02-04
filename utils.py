import os
import pandas as pd
import numpy as np 
from epiweeks import Week, Year
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_scripts.exogenous_datasets import ExogenousDataset 
from data_scripts.endogenous_dataset import EndogenousDataset
from model_scripts.exog_model_utils import transform_to_recurrent
from data_scripts.datasets import GraphData
epideep_column_metadata_file = './data/column_info_epiDeep.json'
column_metadata_file_overlap = './data/column_info_exo_overlap.json'
datainput_file='./data/train_data_weekly_noscale.csv'
histILI_datainput_file='./data/hist_ILI_sorted.csv'
region_graph_input_file="./data/wILI_region_adjacency_list.txt"
debug = False

# code from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def buildNetwork(layers, activation="relu", dropout=0, batchnorm=False):
    """ 
        batchnorm only after first layer
    """
    net = []
    for i in range(1, len(layers)):
        lm = nn.Linear(layers[i-1], layers[i])
        torch.nn.init.xavier_normal_(lm.weight) 
        # batchnorm before activation
        net.append(lm)
        
        if i < len(layers)-1:
            if activation=="relu":
                net.append(nn.ReLU())
            elif activation=="sigmoid":
                net.append(nn.Sigmoid())
            elif activation=="leakyReLU":
                net.append(nn.LeakyReLU())
            if dropout > 0 and i<len(layers)-1:
                net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


def save_results(path_rmse, path_res, predictions, rmse, targets, epiweek, run_no):
            
    df1=pd.DataFrame.from_dict(predictions)
    df2=pd.DataFrame.from_dict(rmse)
    df3=pd.DataFrame.from_dict(targets)

    if not os.path.exists(path_res):
       f = open(path_res, "w")
       f.write('region,epiweek,date,iter_number,val1,val2,val3,val4,pred1,pred2,pred3,pred4'+'\n')
       f.close()
    
    # merge predictions and targets
    df = df3.join(df1)
    #save predictions
    df['epiweek'] = epiweek
    end_week = Week(2020,epiweek)
    end_date = end_week.enddate()
    df['date'] = str(end_date)
    df['iter_number'] = run_no
    cols = list(df.columns)
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df.to_csv(path_res, header=None, index=True, mode='a')


def prepare_train_data(epiweek,k_week_ahead,epideep_seq_len,device):
    """

        @param debug: only works for exog at this moment
    """

    column_metadata_file_overlap = './experiment_setup/feature_module/column_metadata/column_info_all.json'
    column_metadata_ILI_overlap = './experiment_setup/feature_module/column_metadata/column_info_ili.json'
    
    train_end_week = Week(2020,epiweek)
    train_end_date = train_end_week.enddate()
    # start here
    USE_ENDOGENOUS_FEATURES=False
    num_test_weeks = 1
    # this is to get data for epideep in overlap
    # the only difference is in the length of the sequence
    histILI_dataset =EndogenousDataset(epideep_column_metadata_file,histILI_datainput_file,epideep_seq_len,k_week_ahead,\
        str(train_end_date),input_feature_prefix="input_feature",num_test_weeks=num_test_weeks,DEBUG=debug)  # use 5 because TrainDatasets uses that
    dataset =ExogenousDataset(column_metadata_file_overlap,datainput_file,epideep_seq_len,k_week_ahead,training_end_date=str(train_end_date),
                with_endog_features=USE_ENDOGENOUS_FEATURES,input_feature_prefix="input_feature",
                            num_test_weeks=num_test_weeks, DEBUG=debug, endog_scaler=histILI_dataset.scalers)
    dataset_ILI_overlap=ExogenousDataset(column_metadata_ILI_overlap,datainput_file,epideep_seq_len,k_week_ahead,training_end_date=str(train_end_date),
                with_endog_features=USE_ENDOGENOUS_FEATURES,input_feature_prefix="input_feature",
                            num_test_weeks=num_test_weeks, DEBUG=debug, endog_scaler=histILI_dataset.scalers)
    
    region_graph = GraphData(region_graph_input_file)
    clustering_query_length = Variable(torch.from_numpy(histILI_dataset.trainX.values).float()).to(device)
    rnn_data = clustering_query_length.reshape(clustering_query_length.shape[0],clustering_query_length.shape[1],1)
    rnn_label = Variable(torch.from_numpy(histILI_dataset.trainY.values.reshape(-1,1)).float()).to(device)
    clustering_full_length = torch.cat((clustering_query_length,rnn_label),axis=1)
    # test data        
    clustering_query_length_test = Variable(torch.from_numpy(histILI_dataset.testX.values).float()).to(device)
    rnn_data_test = clustering_query_length_test.reshape(clustering_query_length_test.shape[0],clustering_query_length_test.shape[1],1)
    rnn_label_test = Variable(torch.from_numpy(histILI_dataset.testY.values.reshape(-1,1)).float()).to(device)

    
    # Setup Data Loader
    size_feat_input_data = dataset.trainX.values.shape[1]
    # recurrent:
    rec_trainX = transform_to_recurrent(dataset.trainX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
    realX_train=Variable(torch.from_numpy(rec_trainX).float()).to(device)

    rec_testX = transform_to_recurrent(dataset.testX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
    realX_test = Variable(torch.from_numpy(rec_testX).float()).to(device)

    _cat_train = np.concatenate(dataset.trainXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
    catX_train = Variable(torch.from_numpy(_cat_train).float()).to(device)
    realY_train = Variable(torch.from_numpy(dataset.trainY.values).float()).to(device)

    #Convert RegionIDs into integer codes to pass into pytorch batch loader which doesn't accept strings.
    region_ids_train = np.array([int(k.replace("Region","").strip()) if "Region" in k else 11 for k in dataset.trainRegions.values.tolist()])
    regions_train = Variable(torch.from_numpy(region_ids_train)).to(device) 

    # realX_test = Variable(torch.from_numpy(dataset.testX.values).float()).to(device)
    _cat_test = np.concatenate(dataset.testXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
    catX_test = Variable(torch.from_numpy(_cat_test).float()).to(device)
    realY_test = Variable(torch.from_numpy(dataset.testY.values).float()).to(device)
    
    # overlap data for epideep
    clustering_query_length_overlap = Variable(torch.from_numpy(dataset.trainX.values).float()).to(device)[:,5:]
    clustering_query_length_overlap = Variable(torch.from_numpy(dataset_ILI_overlap.trainX.values).float()).to(device)
    rnn_data_overlap = clustering_query_length_overlap.reshape(clustering_query_length_overlap.shape[0],clustering_query_length_overlap.shape[1],1)
    rnn_label_overlap = Variable(torch.from_numpy(dataset.trainY.values.reshape(-1,1)).float()).to(device)
    clustering_full_length_overlap = torch.cat((clustering_query_length_overlap,rnn_label_overlap),axis=1)

    data_loader_hist_train = DataLoader(list(zip(clustering_query_length,clustering_full_length,rnn_data,rnn_label)),shuffle=False,batch_size=dataset.num_regions)
    data_loader_hist_test = DataLoader(list(zip(clustering_query_length_test,rnn_data_test,rnn_label_test)),shuffle=False,batch_size=dataset.num_regions)
    data_loader_train = DataLoader(list(zip(realX_train,catX_train,realY_train,regions_train)),shuffle=False,batch_size=dataset.num_regions)
    data_loader_test = DataLoader(list(zip(realX_test,catX_test,realY_test)),shuffle=False,batch_size=dataset.num_regions)
    data_loader_train_overlap = DataLoader(list(zip(clustering_query_length_overlap,clustering_full_length_overlap,rnn_data_overlap,rnn_label_overlap,\
                                                    realX_train,catX_train,realY_train,regions_train)),shuffle=False,batch_size=dataset.num_regions)
    # return train_datasets, fit_inputs
    return histILI_dataset, data_loader_hist_train, data_loader_hist_test, dataset, region_graph, data_loader_train, data_loader_test, size_feat_input_data, data_loader_train_overlap



if __name__ == "__main__":
    regionName='X'
    epiweek=13
    k_week_ahead=3
    debug=False
    EPIDEEP_SEQ_LEN=5
    device=torch.device('cpu')
    # prepare_train_data(regionName,epiweek,k_week_ahead,EPIDEEP_SEQ_LEN,device)