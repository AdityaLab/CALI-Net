import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import pdb
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .exog_models import GlobalFeatureModule,GlobalRecurrentFeatureModule
import sys
sys.path.append("..")
from data_scripts.exogenous_datasets import ExogenousDataset
from data_scripts.exogenous_datasets_local import ExogenousDatasetLocal
from data_scripts.datasets import GraphData
from model_scripts import exog_models
from collections import OrderedDict

class Model:
    """
        Wrapper function to enable a unified pipline for training, testing,evaluating and plotting results from multiple types of models.
        This function instantiates a model with the requisite hyperparameters and provides a predict function which
        when given input features, returns model outputs.
        Most importantly, it allows the user to instantiate different types of models, just by specifying which models to instantiate in a json file.
    
    """
    def __init__(self,model_parameters_json_file,device):
        with open(model_parameters_json_file) as f:
            self.model_metadata = json.load(f)
        
        #Initialize model with hyperparameters.
        self.model = getattr(exog_models,self.model_metadata['model_type'])(**self.model_metadata['model_hyperparameters']).to(device)
        self.num_epochs = self.model_metadata['NUM_EPOCHS']
        self.laplacian_hyperparameter = self.model_metadata['LAPLACIAN_HYPERPARAMETER']
        self.learning_rate = self.model_metadata['LEARNING_RATE']
        self.region_equity_hyperparameter = self.model_metadata['REGION_EQUITY_HYPERPARAMETER']
        self.ae_reconstruction_weight = self.model_metadata['AE_RECONSTRUCTION_WEIGHT']
        self.region_reconstruction_weight = self.model_metadata['REGION_RECONSTRUCTION_WEIGHT']
        self.kd_hyperparameter = self.model_metadata['KD_HYPERPARAMETER']
        self.l2_reg_hyperparameter = self.model_metadata['L2_REGULARIZATION_HYPERPARAMETER']
    
    def predict(self,batchX,batchCatX):
         return self.model(batchX,batchCatX)
    
    def evaluate(self,predictions,targets,regions,region_col):
        """
            @param predictions: Predictions List.
            @param targets: Targets List.
            @param regions: List object containing region name for each prediction, target instance. Same size as predictions , targets.
            @param region_col: The region_column name
            Return RMSE of the model.
            There are two return values: Overall RMSE, and Per-Region RMSE.
        """
        rmse = lambda x,y: np.sqrt(np.mean(np.square(x - y)))
        
        overall_rmse = rmse(np.array(predictions).ravel(),np.array(targets).ravel())         
        tmp = regions
        tmp = pd.DataFrame(tmp,columns=[region_col])
        tmp['predictions'] = predictions
        tmp['targets'] = targets

        per_region_rmse = OrderedDict()

        for key,val in tmp.groupby(region_col):
            per_region_rmse[key] = rmse(val['predictions'].values.ravel(),val['targets'].values.ravel())
        return overall_rmse,per_region_rmse

    def plot_training_loss(self,training_loss_per_epoch):
        fig,ax=plt.subplots(1,1,figsize=(12,8))
        ax.plot(training_loss_per_epoch)
        ax.set_title("Total Training Loss",fontsize=12)

    def plot(self,predictions,targets,regions,dates):
        """
            @param predictions: Predictions Series object.
            @param targets: Targets Series object.
            @param regions: Regions Series object.
            @param dates: Dates Series Object.
            
            This function will plot training and testing time-series predictions.
        """
        pass

def train_with_laplacian_reg(data_loader_train,model,laplacian_graph,device):
    """
          @param laplacian_graph: A dictionary where each region has a specific key and value is the corresponding row of the Laplacian for that region.

    """
    #Hyperparameters
    NUMEPOCHS=model.num_epochs
    LEARNING_RATE=model.learning_rate
    _lambda=model.laplacian_hyperparameter
    _beta=model.region_equity_hyperparameter

    #Instantiate Loss and Optimizer.
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.model.parameters(),lr=model.learning_rate,weight_decay=model.l2_reg_hyperparameter)
    
    #Bookkeeping Structures
    wILI_loss_per_epoch=list()
    total_loss_per_epoch=list()
    embedding_loss_per_epoch=list()
    embeddings_per_region=dict() #Capture Embeddings Per Region At The Final Epoch
    region_equity_loss_per_epoch=list()
    lapreg_loss_per_epoch=list()

    for epoch in range(NUMEPOCHS):
        
        wILI_loss_per_batch=list()
        embedding_loss_per_batch=list()
        total_loss_per_batch=list()
        region_equity_loss_per_batch=list()
        lap_reg_loss_per_batch=list()
        
        #Record predictions and actual values only from the last epoch.
        preds=list()
        actual_values=list()
        region_vals=list()
        real_inputs=list()
        
        
        for batchXReal,batchXCat,batchY,batchRegions in data_loader_train:
            optim.zero_grad()
            wILI_preds,wILI_embedding,region_encoding_reconstructed,region_embedding = model.predict(batchXReal,batchXCat)
            
            wILI_loss = criterion(wILI_preds.squeeze(),batchY.squeeze())
            embedding_reconstruction_loss = criterion(region_encoding_reconstructed,batchXCat)
            ################ Region Equity ################
            region_equity=torch.nn.functional.mse_loss(wILI_preds.squeeze(), batchY.squeeze(), reduction='none')
            region_equity = (region_equity.squeeze().unsqueeze(1) - region_equity.squeeze()).flatten()

            region_equity = torch.pow(region_equity,2) 
            region_equity = torch.sum(region_equity)
            ############## Region Equity End. #############
             
            ############## Laplacian Reg. #################

            lap_reg = 0.0 #Initialize Laplacian Regularization Loss to 0.0

	    #Stack such that each row of the lap_mat variable corresponds to the row in the laplacian matrix per region.
            lap_mat = np.vstack([laplacian_graph[r] for r in batchRegions.cpu().data.numpy().tolist()])
            lap_mat = Variable(torch.from_numpy(lap_mat).float()).to(device) 
            lap_reg = torch.matmul(torch.transpose(region_embedding,0,1),lap_mat)
            lap_reg =  torch.matmul(lap_reg,region_embedding)
            _eye = torch.eye(lap_reg.size(0)).to(device)  #Create identity mat. and move to `device`. Used for Hadamard prod. with lap_reg for trace calc. 
            lap_reg = torch.sum(_eye*lap_reg).pow(2)  #Square the sum.  (Formulation, everything but the square is from: (Climate Multi-model Regression Using Spatial Smoothing))
            
            ############## Laplacian Reg. End #############

            total_loss = wILI_loss + _lambda*embedding_reconstruction_loss + 0.1*lap_reg

            total_loss_per_batch.append(total_loss.item())
            embedding_loss_per_batch.append(embedding_reconstruction_loss.item())
            wILI_loss_per_batch.append(wILI_loss.item())
            lap_reg_loss_per_batch.append(_beta*lap_reg.item())
            region_equity_loss_per_batch.append(0*region_equity.item())
            
            #Append Predictions and Targets
            preds.extend(wILI_preds.cpu().data.numpy().ravel().tolist())
            actual_values.extend(batchY.cpu().data.numpy().ravel().tolist())
            real_inputs.append(batchXReal.cpu().data.numpy())
            
            #Backpropagate and Update model
            total_loss.backward()
            optim.step()

        wILI_loss_per_epoch.append(np.mean(wILI_loss_per_batch))
        embedding_loss_per_epoch.append(np.mean(embedding_loss_per_batch))

        total_loss_per_epoch.append(np.mean(total_loss_per_batch))
        region_equity_loss_per_epoch.append(np.mean(region_equity_loss_per_batch))
        lapreg_loss_per_epoch.append(np.mean(lap_reg_loss_per_batch))
    
    fig,ax=plt.subplots(1,1,figsize=(12,8))
    ax.plot(wILI_loss_per_epoch,c='b')
    ax.plot(region_equity_loss_per_epoch,c='g')
    ax.plot(total_loss_per_epoch,c='r')
    ax.plot(lapreg_loss_per_epoch,c='k')
    ax.legend(['wILI_Loss','Region_Equity_Loss','Total Loss','Lap. Reg. Loss'],fontsize=16)
    ax.set_title("Training Losses",fontsize=16)
    
    real_inputs = np.concatenate(real_inputs)
    return total_loss_per_epoch,preds,actual_values,real_inputs


def train_local(data_loader_train,model,device):
    
    #Hyperparameters
    NUMEPOCHS=model.num_epochs
    LEARNING_RATE=model.learning_rate
    _lambda=model.laplacian_hyperparameter  #Here it is used only for the reconstruction loss.

    #Instantiate Loss and Optimizer.
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.model.parameters(),lr=model.learning_rate,weight_decay=model.l2_reg_hyperparameter)
    
    #Bookkeeping Structures
    wILI_loss_per_epoch=list()
    total_loss_per_epoch=list()
    embedding_loss_per_epoch=list()
    embeddings_per_region=dict() #Capture Embeddings Per Region At The Final Epoch
    
    for epoch in range(NUMEPOCHS):
        
        wILI_loss_per_batch=list()
        embedding_loss_per_batch=list()
        total_loss_per_batch=list()
        
        #Record predictions and actual values only from the last epoch.
        preds=list()
        actual_values=list()
        region_vals=list()
        real_inputs=list()
        
        
        for batchXReal,batchXCat,batchY in data_loader_train:
            optim.zero_grad()
            wILI_preds,wILI_embedding,region_encoding_reconstructed,region_embedding = model.predict(batchXReal,batchXCat)
            
            wILI_loss = criterion(wILI_preds.squeeze(),batchY.squeeze())
            embedding_reconstruction_loss = criterion(region_encoding_reconstructed,batchXCat)

            total_loss = wILI_loss + _lambda*embedding_reconstruction_loss 

            total_loss_per_batch.append(total_loss.item())
            embedding_loss_per_batch.append(embedding_reconstruction_loss.item())
            wILI_loss_per_batch.append(wILI_loss.item())
            
            #Append Predictions and Targets
            preds.extend(wILI_preds.cpu().data.numpy().ravel().tolist())
            actual_values.extend(batchY.cpu().data.numpy().ravel().tolist())
            real_inputs.append(batchXReal.cpu().data.numpy())
            
            #Backpropagate and Update model
            total_loss.backward()
            optim.step()

        wILI_loss_per_epoch.append(np.mean(wILI_loss_per_batch))
        embedding_loss_per_epoch.append(np.mean(embedding_loss_per_batch))

        total_loss_per_epoch.append(np.mean(total_loss_per_batch))
    
    
    fig,ax=plt.subplots(1,1,figsize=(12,8))
    ax.plot(wILI_loss_per_epoch,c='b')
    ax.plot(total_loss_per_epoch,c='r')
    ax.legend(['wILI_Loss','Total Loss'],fontsize=16)
    ax.set_title("Training Losses",fontsize=16)
    
    real_inputs = np.concatenate(real_inputs)
    return total_loss_per_epoch,preds,actual_values,real_inputs



def train(data_loader_train,model,device):
    
    #Hyperparameters
    NUMEPOCHS=model.num_epochs
    LEARNING_RATE=model.learning_rate
    _lambda=model.laplacian_hyperparameter
    _beta=model.region_equity_hyperparameter

    #Instantiate Loss and Optimizer.
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.model.parameters(),lr=model.learning_rate,weight_decay=model.l2_reg_hyperparameter)
    
    #Bookkeeping Structures
    wILI_loss_per_epoch=list()
    total_loss_per_epoch=list()
    embedding_loss_per_epoch=list()
    embeddings_per_region=dict() #Capture Embeddings Per Region At The Final Epoch
    region_equity_loss_per_epoch=list()
    
    for epoch in range(NUMEPOCHS):
        
        wILI_loss_per_batch=list()
        embedding_loss_per_batch=list()
        total_loss_per_batch=list()
        region_equity_loss_per_batch=list()
        
        #Record predictions and actual values only from the last epoch.
        preds=list()
        actual_values=list()
        region_vals=list()
        real_inputs=list()
        
        
        for batchXReal,batchXCat,batchY in data_loader_train:
            optim.zero_grad()
            wILI_preds,wILI_embedding,region_encoding_reconstructed,region_embedding = model.predict(batchXReal,batchXCat)
            
            wILI_loss = criterion(wILI_preds.squeeze(),batchY.squeeze())
            embedding_reconstruction_loss = criterion(region_encoding_reconstructed,batchXCat)
            region_equity=torch.nn.functional.mse_loss(wILI_preds.squeeze(), batchY.squeeze(), reduction='none')
            region_equity = (region_equity.squeeze().unsqueeze(1) - region_equity.squeeze()).flatten()

            region_equity = torch.pow(region_equity,2) 
            region_equity = torch.sum(region_equity)

            total_loss = wILI_loss + _lambda*embedding_reconstruction_loss + _beta*region_equity

            total_loss_per_batch.append(total_loss.item())
            embedding_loss_per_batch.append(embedding_reconstruction_loss.item())
            wILI_loss_per_batch.append(wILI_loss.item())
            region_equity_loss_per_batch.append(_beta*region_equity.item())
            
            #Append Predictions and Targets
            preds.extend(wILI_preds.cpu().data.numpy().ravel().tolist())
            actual_values.extend(batchY.cpu().data.numpy().ravel().tolist())
            real_inputs.append(batchXReal.cpu().data.numpy())
            
            #Backpropagate and Update model
            total_loss.backward()
            optim.step()

        wILI_loss_per_epoch.append(np.mean(wILI_loss_per_batch))
        embedding_loss_per_epoch.append(np.mean(embedding_loss_per_batch))

        total_loss_per_epoch.append(np.mean(total_loss_per_batch))
        region_equity_loss_per_epoch.append(np.mean(region_equity_loss_per_batch))
    
    
    fig,ax=plt.subplots(1,1,figsize=(12,8))
    ax.plot(wILI_loss_per_epoch,c='b')
    ax.plot(region_equity_loss_per_epoch,c='g')
    ax.plot(total_loss_per_epoch,c='r')
    ax.legend(['wILI_Loss','Region_Equity_Loss','Total Loss'],fontsize=16)
    ax.set_title("Training Losses",fontsize=16)
    
    real_inputs = np.concatenate(real_inputs)
    return total_loss_per_epoch,preds,actual_values,real_inputs


def test(data_loader_test,model,training_hyperparameters=None):
    """
          @param data_loader_test: The data loader to be used for the testing set.
          @param model: The initial trained model (on the initial training set).
          @param adaptive_retraining: Boolean flag indicating if the training is to be done in a batch fashion or an adaptive fashion. If adaptive, then 
                                      training_hyperparameters need to be specified. Adaptive retraining entails retraining the initial model with incrementally
                                      larger data i.e one additional instance from the test set is incorporated back into the training set each time (i.e after 
                                      each testing cycle and corresponding calculation of RMSE has concluded). This is simulating the way the data arrives i.e 
                                      one data point each week of wILI, COVID etc. that is incorporated during the model re-training process.
    """

    preds=list()
    targets=list()
    region_vals=list()
    real_inputs=list()
    for batchXReal,batchXCat,batchY in data_loader_test:
        wILI_preds,wILI_embedding,_,_ = model.predict(batchXReal,batchXCat)
        preds.extend(wILI_preds.cpu().data.numpy().ravel().tolist())
        targets.extend(batchY.cpu().data.numpy().ravel().tolist())
        real_inputs.append(batchXReal.cpu().data.numpy())
        
    real_inputs = np.concatenate(real_inputs)

    return preds,targets,real_inputs

def transform_to_recurrent(arr,sequence_length,num_features):
    """
        @param arr: The 2D numpy array to be transformed to a 3D array of size 
                    (arr.shape[0],sequence_length,num_features). This format is amenable
                    for use with a recurrent network architecture.
        @param sequence_length: This is technically the number of steps for which the recurrence will be unrolled.
        @param num_features: The number of exogenous features we are planning to use.
    """
    start_idx=0
    ret_arr=np.zeros((arr.shape[0],sequence_length,num_features))
    for feat_idx in range(num_features):
        ret_arr[:,:,feat_idx] = arr[:,start_idx:start_idx+sequence_length]
        start_idx=start_idx+sequence_length

    return ret_arr

def execute_local(training_hyperparameters,previous_model=None,recurrent=False):
    """
           @param recurrent: A flag which indicates whether the model to be trained is recurrent in nature or not. If so, the trainX and testX arrays
                               are transformed to be of type (batch_size,sequence_length,feature_size). This change happens in the Exogenous (and Endogenous) dataset classes.
    """
    #Unpack Hyperparameters
    model_hyperparameters_file=training_hyperparameters['model_hyperparameters_file']
    training_end_date=training_hyperparameters['training_end_date']
    column_metadata_file=training_hyperparameters['column_metadata_file']
    datainput_file=training_hyperparameters['datainput_file']
    experiment_output_dir=training_hyperparameters['experiment_output_dir']
    max_hist=training_hyperparameters['max_hist']
    k_week_ahead=training_hyperparameters['k_week_ahead']
    USE_ENDOGENOUS_FEATURES=training_hyperparameters['USE_ENDOGENOUS_FEATURES']
    NUMEXPERIMENTS=training_hyperparameters['NUMEXPERIMENTS']
    device=training_hyperparameters['device']

    #Create N separate data-input files, each file containing all data per region. This can then be used to 
    dataset=ExogenousDatasetLocal(column_metadata_file,datainput_file,max_hist,k_week_ahead,training_end_date=training_end_date,
            with_endog_features=USE_ENDOGENOUS_FEATURES,input_feature_prefix="input_feature")     
    
    ##########
    models=list()
    preds_all=list()
    tgts_all=list()
    regions_all=list()   
    dates_all=list()     
    train_preds_all=list()
    train_targets_all=list()
    train_regions_all=list()
    train_dates_all=list()

    for i in range(NUMEXPERIMENTS): 
        if previous_model is None:
            #Re-Instantiate Model if previous_model isn't supplied.
            model = Model(model_hyperparameters_file,device)
            #Update Model.state_dict from a saved pre-trained global model.
        
        else:
            model = previous_model
        
        # Setup Data Loader

        if recurrent:
            rec_trainX = transform_to_recurrent(dataset.trainX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
            realX_train=Variable(torch.from_numpy(rec_trainX).float()).to(device)

            rec_testX = transform_to_recurrent(dataset.testX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
            realX_test = Variable(torch.from_numpy(rec_testX).float()).to(device)
        else:
            realX_train = Variable(torch.from_numpy(dataset.trainX.values).float()).to(device)
            realX_test = Variable(torch.from_numpy(dataset.testX.values).float()).to(device)

        _cat_train = np.concatenate(dataset.trainXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
        catX_train = Variable(torch.from_numpy(_cat_train).float()).to(device)
        realY_train = Variable(torch.from_numpy(dataset.trainY.values).float()).to(device)

        ##Convert RegionIDs into integer codes to pass into pytorch batch loader which doesn't accept strings.
        #region_ids_train = np.array([int(k.replace("Region","").strip()) if "Region" in k else 11 for k in dataset.trainRegions.values.tolist()])
        #regions_train = Variable(torch.from_numpy(region_ids_train)).to(device) 
                
        _cat_test = np.concatenate(dataset.testXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
        catX_test = Variable(torch.from_numpy(_cat_test).float()).to(device)
        realY_test = Variable(torch.from_numpy(dataset.testY.values).float()).to(device)
 
        data_loader_train = DataLoader(list(zip(realX_train,catX_train,realY_train)),shuffle=False,batch_size=dataset.num_regions)
        data_loader_test = DataLoader(list(zip(realX_test,catX_test,realY_test)),shuffle=False,batch_size=dataset.num_regions)
        #Training
        total_loss_per_epoch,preds,actual_values,real_inputs = train_local(data_loader_train,model,device=device)


                
        print("Train End Date = {}".format(training_end_date))
        #print("Train Inputs Hist WILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3]))) 
        print("Train Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3])))
        print("Train All INPUTS MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs),np.min(real_inputs),np.mean(real_inputs)))
        print("Train Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(preds),np.min(preds),np.mean(preds))) 
        
        #Inverse Transform Train Predictions
        inv_trans_df_train = dataset.inverse_transform(preds,actual_values,dataset.trainRegions,dataset.target_col,transform_targets=True)
        preds = inv_trans_df_train['inv_transformed_predictions'].values.tolist()
        tgts = inv_trans_df_train['inv_transformed_targets'].values.tolist() 
        reg = inv_trans_df_train[dataset.region_col].values.tolist()
        
        train_preds_all.append(preds)
        train_targets_all.append(tgts)
        train_regions_all.append(reg)
        train_dates_all.append(dataset.data.loc[dataset.train_indices][dataset.date_col].apply(lambda x: dt.strftime(x,"%Y-%m-%d")).values.tolist())
        
        #=================================== Evaluation ======================================#
        #Testing
        test_preds,test_targets,real_inputs=test(data_loader_test,model)
        print("Test Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3]))) 
        print("Test Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(test_preds),np.min(test_preds),np.mean(test_preds)))
        
        #Inverse Transform
        inv_trans_df_test = dataset.inverse_transform(test_preds,test_targets,dataset.testRegions,\
        				     dataset.target_col,transform_targets=False)
        
        preds = inv_trans_df_test['inv_transformed_predictions'].values.tolist()
        tgts = test_targets
        reg = inv_trans_df_test[dataset.region_col].values.tolist()
        
        #Append Predictions,Targets,Regions,Dates and the trained model.
        preds_all.append(preds)
        tgts_all.append(tgts)
        regions_all.append(reg)
        
        dates_all.append(dataset.data.loc[dataset.test_indices][dataset.date_col].apply(lambda x: dt.strftime(x,"%Y-%m-%d")).values.tolist())
        models.append(model)
  
    return train_preds_all,train_targets_all,train_regions_all,train_dates_all,preds_all,tgts_all,regions_all,dates_all,models
   


def execute(training_hyperparameters,previous_model=None,recurrent=False,laplacian_regularization=False):
    """ 
             @param recurrent: A flag which indicates whether the model to be trained is 
                               recurrent in nature or not. If so, the trainX and testX arrays
                               are transformed to be of type (batch_size,sequence_length,feature_size). This change happens in the Exogenous (and Endogenous) dataset classes
             @param laplacian_regularization: A flag which indicates whether or not to use th Laplacian regularization constraint. Note: If true, we call a different train() function.

    """ 
    #Unpack Hyperparameters
    model_hyperparameters_file=training_hyperparameters['model_hyperparameters_file']
    training_end_date=training_hyperparameters['training_end_date']
    column_metadata_file=training_hyperparameters['column_metadata_file']
    datainput_file=training_hyperparameters['datainput_file']
    experiment_output_dir=training_hyperparameters['experiment_output_dir']
    max_hist=training_hyperparameters['max_hist']
    k_week_ahead=training_hyperparameters['k_week_ahead']
    USE_ENDOGENOUS_FEATURES=training_hyperparameters['USE_ENDOGENOUS_FEATURES']
    region_graph_input_file=training_hyperparameters['region_graph_input_file']
    NUMEXPERIMENTS=training_hyperparameters['NUMEXPERIMENTS']
    device=training_hyperparameters['device']

    #Create N separate data-input files, each file containing all data per region. This can then be used to 
    dataset=ExogenousDataset(column_metadata_file,datainput_file,max_hist,k_week_ahead,training_end_date=training_end_date,
            with_endog_features=USE_ENDOGENOUS_FEATURES,input_feature_prefix="input_feature")     
    
    region_graph = GraphData(region_graph_input_file)   
    
    ##########
    models=list()
    preds_all=list()
    tgts_all=list()
    regions_all=list()   
    dates_all=list()     
    train_preds_all=list()
    train_targets_all=list()
    train_regions_all=list()
    train_dates_all=list()

    for i in range(NUMEXPERIMENTS): 
        if previous_model is None:
            #Re-Instantiate Model if previous_model isn't supplied.
            model = Model(model_hyperparameters_file,device)
        
        else:
            model = previous_model
        
        # Setup Data Loader

        if recurrent:
            rec_trainX = transform_to_recurrent(dataset.trainX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
            realX_train=Variable(torch.from_numpy(rec_trainX).float()).to(device)

            rec_testX = transform_to_recurrent(dataset.testX.values,sequence_length=dataset.max_hist,num_features=len(dataset.feature_cols))
            realX_test = Variable(torch.from_numpy(rec_testX).float()).to(device)
        else:
            realX_train = Variable(torch.from_numpy(dataset.trainX.values).float()).to(device)
            realX_test = Variable(torch.from_numpy(dataset.testX.values).float()).to(device)

        _cat_train = np.concatenate(dataset.trainXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
        catX_train = Variable(torch.from_numpy(_cat_train).float()).to(device)
        realY_train = Variable(torch.from_numpy(dataset.trainY.values).float()).to(device)

        #Convert RegionIDs into integer codes to pass into pytorch batch loader which doesn't accept strings.
        region_ids_train = np.array([int(k.replace("Region","").strip()) if "Region" in k else 11 for k in dataset.trainRegions.values.tolist()])
        regions_train = Variable(torch.from_numpy(region_ids_train)).to(device) 
                
        _cat_test = np.concatenate(dataset.testXCat.apply(lambda x: np.array(x)[:,None]).values,axis=1).T
        catX_test = Variable(torch.from_numpy(_cat_test).float()).to(device)
        realY_test = Variable(torch.from_numpy(dataset.testY.values).float()).to(device)
 
        if laplacian_regularization:
            data_loader_train = DataLoader(list(zip(realX_train,catX_train,realY_train,regions_train)),shuffle=False,batch_size=dataset.num_regions)
            data_loader_test = DataLoader(list(zip(realX_test,catX_test,realY_test)),shuffle=False,batch_size=dataset.num_regions)

            #Call new train() method.
            total_loss_per_epoch,preds,actual_values,real_inputs = train_with_laplacian_reg(data_loader_train,model,region_graph.laplacian_dict,device)
        else:
            data_loader_train = DataLoader(list(zip(realX_train,catX_train,realY_train)),shuffle=False,batch_size=dataset.num_regions)
            data_loader_test = DataLoader(list(zip(realX_test,catX_test,realY_test)),shuffle=False,batch_size=dataset.num_regions)
            #Training
            total_loss_per_epoch,preds,actual_values,real_inputs = train(data_loader_train,model,device=device)


                
        print("Train End Date = {}".format(training_end_date))
        #print("Train Inputs Hist WILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3]))) 
        print("Train Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3])))
        print("Train All INPUTS MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs),np.min(real_inputs),np.mean(real_inputs)))
        print("Train Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(preds),np.min(preds),np.mean(preds))) 
        
        #Inverse Transform Train Predictions
        inv_trans_df_train = dataset.inverse_transform(preds,actual_values,dataset.trainRegions,dataset.target_col,transform_targets=True)
        preds = inv_trans_df_train['inv_transformed_predictions'].values.tolist()
        tgts = inv_trans_df_train['inv_transformed_targets'].values.tolist() 
        reg = inv_trans_df_train[dataset.region_col].values.tolist()
        
        train_preds_all.append(preds)
        train_targets_all.append(tgts)
        train_regions_all.append(reg)
        train_dates_all.append(dataset.data.loc[dataset.train_indices][dataset.date_col].apply(lambda x: dt.strftime(x,"%Y-%m-%d")).values.tolist())
        
        #=================================== Evaluation ======================================#
        #Testing
        test_preds,test_targets,real_inputs=test(data_loader_test,model)
        print("Test Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3]))) 
        print("Test Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(test_preds),np.min(test_preds),np.mean(test_preds)))
        
        #Inverse Transform
        inv_trans_df_test = dataset.inverse_transform(test_preds,test_targets,dataset.testRegions,\
        				     dataset.target_col,transform_targets=False)
        
        preds = inv_trans_df_test['inv_transformed_predictions'].values.tolist()
        tgts = test_targets
        reg = inv_trans_df_test[dataset.region_col].values.tolist()
        
        #Append Predictions,Targets,Regions,Dates and the trained model.
        preds_all.append(preds)
        tgts_all.append(tgts)
        regions_all.append(reg)
        
        dates_all.append(dataset.data.loc[dataset.test_indices][dataset.date_col].apply(lambda x: dt.strftime(x,"%Y-%m-%d")).values.tolist())
        models.append(model)
  
    return train_preds_all,train_targets_all,train_regions_all,train_dates_all,preds_all,tgts_all,regions_all,dates_all,models


