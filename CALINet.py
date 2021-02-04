
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
import pandas as pd
import pickle; import os
from EpiDeepCN import EpiDeepCN
from utils import buildNetwork, prepare_train_data, save_results
from model_scripts.exog_model_utils import Model
from rnnAttention import RNN
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # tested only in cpu
dtype = torch.float
EPIDEEP_SEQ_LEN=5
POST_TRAIN_EPOCHS = 100
MODEL_HYPERPARAMS = "./experiment_setup/feature_module/model_specifications/global_recurrent_feature_model.json"
pretrain_epochs=200

def TrainPredict(currentWeek,epiweek,run_no=0):  
    """
        #Here are the inputs example
        #currentWeek=27
    """
    print(' week', epiweek)
    suffix = '_'
    
    # print(suffix)
    predictions={}
    targets={}
    rmse={}
    # will predict K ahead
    k_week_ahead = 1; K=4
    while k_week_ahead <=K: 
        print("====== next", k_week_ahead)

        _, data_loader_hist_train,\
            _, dataset, region_graph,\
                data_loader_train, data_loader_test, _,\
                    data_loader_train_overlap = \
                    prepare_train_data(epiweek,k_week_ahead,EPIDEEP_SEQ_LEN,device)

        calinet = CALINet()
        # pretrain epideep-cn
        calinet.pretrain_epideep(data_loader_hist_train)
        # train calinet for the corresponding k_week_ahead
        preds,actual_values,real_inputs =\
            calinet.train(data_loader_train,data_loader_hist_train,data_loader_train_overlap,region_graph.laplacian_dict)
        
        print("Train End Date = {}".format(epiweek))
        print("Train Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:5]),np.min(real_inputs[:,:5]),np.mean(real_inputs[:,:5])))
        print("Train Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(preds),np.min(preds),np.mean(preds))) 
        
        #=================================== Evaluation ======================================#
        inv_trans_df = dataset.inverse_transform(preds,actual_values,dataset.trainRegions,\
                                            dataset.target_col,transform_targets=True)
        
        preds = inv_trans_df['inv_transformed_predictions'].values.tolist()
        tgts = inv_trans_df['inv_transformed_targets'].values.tolist()
        reg = inv_trans_df[dataset.region_col].values.tolist()

        overall,per_region_rmse = calinet.feat_module.evaluate(predictions=preds,targets=tgts,regions=reg,region_col=dataset.region_col)
        
        print("Train RMSE Overall = {}, Per Region = {}".format(overall,per_region_rmse))
        #Testing
        test_preds,test_targets,real_inputs=calinet.test(data_loader_test)
        print("Test Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:5]),np.min(real_inputs[:,:5]),np.mean(real_inputs[:,:5]))) 
        print("Test Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(test_preds),np.min(test_preds),np.mean(test_preds)))

        #Evaluation in test
        
        #Inverse Transform
        inv_trans_df_test = dataset.inverse_transform(test_preds,test_targets,dataset.testRegions,\
                                                    dataset.target_col,transform_targets=False)
        preds = inv_trans_df_test['inv_transformed_predictions'].values.tolist()
        tgts = test_targets
        reg = inv_trans_df_test[dataset.region_col].values.tolist()

        overall_test,per_region_rmse_test = calinet.feat_module.evaluate(predictions=preds,\
                                targets=tgts,regions=reg,region_col=dataset.region_col)
        
        print("Test RMSE Overall = {}, Per Region = {}".format(overall_test,per_region_rmse_test))
        print(inv_trans_df_test)
        
        def get_results_dict(predictions,targets,regions,region_col):
            """
                @param predictions: Predictions List.
                @param targets: Targets List.
                @param regions: List object containing region name for each prediction, target instance. Same size as predictions , targets.
                @param region_col: The region_column name
                Return RMSE of the model.
                There are two return values: Overall RMSE, and Per-Region RMSE.
            """
            rmse = lambda x,y: np.sqrt(np.nanmean(np.square(x - y)))
            tmp = regions
            tmp = pd.DataFrame(tmp,columns=[region_col])
            tmp['predictions'] = predictions
            tmp['targets'] = targets

            per_region_rmse = OrderedDict()
            per_region_pred = OrderedDict()
            per_region_target = OrderedDict()

            for key,val in tmp.groupby(region_col):
                per_region_rmse[key] = rmse(val['predictions'].values.ravel(),val['targets'].values.ravel())
                per_region_pred[key] = val['predictions'].values.ravel().item()
                per_region_target[key] = val['targets'].values.ravel().item()
            return per_region_pred, per_region_rmse, per_region_target

        predictions['pred'+str(k_week_ahead)], rmse['rmse'+str(k_week_ahead)],\
            targets['val'+str(k_week_ahead)] = get_results_dict(preds,tgts,reg,dataset.region_col)

        k_week_ahead += 1

    for k_week_ahead in range(K+1,5):
        predictions['pred'+str(k_week_ahead)], rmse['rmse'+str(k_week_ahead)],\
            targets['val'+str(k_week_ahead)] = get_results_dict(np.full(len(preds), np.nan),np.full(len(tgts),np.nan),reg,dataset.region_col)

    # epiglobal for epideep + global
    path_rmse = './rmse_results/rmse_'+suffix+'.csv'
    path_res = './rmse_results/results_'+suffix+'.csv'
    save_results(path_rmse, path_res, predictions, rmse, targets, epiweek, run_no)


class CALINet(nn.Module):
    def __init__(self):
        super(CALINet, self).__init__()
        # create and pre-train epideep
        self.epideep = EpiDeepCN(EPIDEEP_SEQ_LEN, 20, EPIDEEP_SEQ_LEN+1, 20, 4, device=device)
        
        self.module_g = buildNetwork([40,16]).to(device)
        self.module_h = buildNetwork([32,16]).to(device)
        self.module_f1 = buildNetwork([16,16,16]).to(device)
        # reconstruct embedding
        self.module_g_prime = buildNetwork([16,40]).to(device)
        self.module_h_prime = buildNetwork([16,32]).to(device)
        self.module_f2 = buildNetwork([16,1]).to(device)
        # create CAEM module
        self.feat_module = Model(MODEL_HYPERPARAMS,device)


    def pretrain_epideep(self,data_loader_hist_train):
        self.epideep.fit_with_dataloader(data_loader_hist_train,num_epoch=pretrain_epochs)
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
            def forward(self, x):
                return x
        # replace decoder by identity
        self.epideep.decoder = Identity()
        self.epideep.regressor = Identity()

    def test(
            self,
            data_loader_test
        ):
        preds=list()
        targets=list()
        real_inputs=list()

        self.feat_module.model.eval()
        self.module_f1.eval()
        self.module_f2.eval()
        self.module_h.eval()

        for batchXReal,batchXCat,batchY in data_loader_test:
            _, wILI_embedding,_,_ = self.feat_module.predict(batchXReal,batchXCat)
            out = self.module_h.forward(wILI_embedding)
            out = self.module_f1.forward(out)
            preds_target_model = self.module_f2.forward(out)
            preds.extend(preds_target_model.cpu().data.numpy().ravel().tolist())
            targets.extend(batchY.cpu().data.numpy().ravel().tolist())
            real_inputs.append(batchXReal.cpu().data.numpy())
            
        real_inputs = np.concatenate(real_inputs)
        return preds,targets,real_inputs


    def train(
            self,
            data_loader_train,
            data_loader_hist_train,
            data_loader_train_overlap,
            laplacian_graph,
        ):
        # load hyperparameters
        NUMEPOCHS=350 
        NUMEPOCHS=self.feat_module.num_epochs
        _beta=self.feat_module.laplacian_hyperparameter  # this is for reconstruction
        _recon_weight= self.feat_module.ae_reconstruction_weight # input reconstruction
        _lambda =  self.feat_module.region_reconstruction_weight # region embedding reconstruction
        _alpha = self.feat_module.kd_hyperparameter

        # loss and optimizer.
        criterion = nn.MSELoss()
        params = list(self.feat_module.model.parameters()) + list(self.parameters())

        """
            Alternating training
        """
        # make the batches real
        list_batches_feat_module_overlap = \
            [(a,b,c,d, batchXReal,batchXCat,batchY,batchRegions) for (a,b,c,d, batchXReal,batchXCat,batchY,batchRegions) in data_loader_train_overlap] 
        list_batches_epideep = [(a,b,c,d) for (a,b,c,d) in data_loader_hist_train] 
        
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=self.feat_module.learning_rate,
            weight_decay=self.feat_module.l2_reg_hyperparameter
        )
        for _ in range(NUMEPOCHS):
            
            total_loss_per_batch=list()
            #Record predictions and actual values only from the last epoch.
            
            total_loss = torch.tensor([0.], dtype=dtype, device=device)
            self.module_g.train()
            for param in self.module_g.parameters():
                param.requires_grad = True
            
            source_loss = torch.tensor([0.], dtype=dtype, device=device)
            for _ in range(5):
                # source model: do a full pass with epideep 
                batch_idx = np.random.choice(len(list_batches_epideep), 1).item()
                batch_clustering_query_length,_,batch_rnn_data,batch_rnn_label = list_batches_epideep[batch_idx]  
                epi_emb = self.epideep.predict(batch_clustering_query_length,batch_rnn_data)
                out = self.module_g.forward(epi_emb)
                out = self.module_f1.forward(out)
                source_reconstructed = self.module_g_prime.forward(out)
                preds_source_model = self.module_f2.forward(out)
                
                #Backpropagate and Update model
                iter_source_loss = criterion(preds_source_model.squeeze(),batch_rnn_label.squeeze())
                iter_source_recon_loss = criterion(source_reconstructed.squeeze(),epi_emb.squeeze())
                source_loss += iter_source_loss + _recon_weight*iter_source_recon_loss
            optim.zero_grad()
            source_loss.backward()
            optim.step()

            self.module_g.eval()
            for param in self.module_g.parameters():
                param.requires_grad = False

            # predict in the whole dataset to get nu
            self.module_g.eval()
            self.module_f1.eval()
            self.module_f2.eval()
            _max=-1*np.inf; _min=np.inf
            for clustering_query_length,_,rnn_data,_,_,_,Y,_ in list_batches_feat_module_overlap:
                out = self.module_g.forward(self.epideep.predict(clustering_query_length,rnn_data))
                out = self.module_f1.forward(out)
                preds_source_model = self.module_f2.forward(out)
                source_error_all_data = F.mse_loss(preds_source_model.squeeze(),Y.squeeze(),reduction='none')
                if source_error_all_data.max() > _max:
                    _max = source_error_all_data.max()
                if source_error_all_data.min() < _min:
                    _min = source_error_all_data.min()
            nu = _max - _min

            for _ in range(2): # train with target model 2 times more than source model
                target_loss = torch.tensor([0.], dtype=dtype, device=device, requires_grad=True)
                for _ in range(5):
                    batch_idx = np.random.choice(len(list_batches_feat_module_overlap), 1).item()
                    batch_clustering_query_length,_,batch_rnn_data,batch_rnn_label,\
                        batchXReal,batchXCat,batchY,batchRegions = list_batches_feat_module_overlap[batch_idx]
                    # to turn off dropout and batchnorm_recon_weight
                    self.module_g.eval()
                    self.module_f1.eval()  
                    self.module_f2.eval()
                    hint_source_emb = self.module_g.forward(self.epideep.predict(batch_clustering_query_length,batch_rnn_data))
                    out = self.module_f1.forward(hint_source_emb)
                    # make source output static
                    hint_source_emb = torch.tensor(hint_source_emb.data, requires_grad=False)
                    preds_source_model = self.module_f2.forward(out)
                    self.module_g.train()
                    self.module_f1.train()
                    self.module_f2.train()
                    _,wILI_embedding,region_encoding_reconstructed,region_embedding = self.feat_module.predict(batchXReal,batchXCat)
                    hint_target_emb = self.module_h.forward(wILI_embedding)
                    out = self.module_f1.forward(hint_target_emb)
                    target_reconstructed = self.module_h_prime.forward(out)
                    preds_target_model = self.module_f2.forward(out)
                    region_embedding_reconstruction_loss = criterion(region_encoding_reconstructed,batchXCat)
                    iter_target_recon_loss = criterion(target_reconstructed.squeeze(),wILI_embedding.squeeze())

                    ############## Laplacian Reg. #################
                    
                    #Stack such that each row of the lap_mat variable corresponds to the row in the laplacian matrix per region.
                    lap_mat = np.vstack([laplacian_graph[r] for r in batchRegions.cpu().data.numpy().tolist()])
                    lap_mat = Variable(torch.from_numpy(lap_mat).float()).to(device) 
                    lap_reg = torch.matmul(torch.transpose(region_embedding,0,1),lap_mat)
                    lap_reg =  torch.matmul(lap_reg,region_embedding)
                    _eye = torch.eye(lap_reg.size(0)).to(device)  #Create identity mat. and move to `device`. Used for Hadamard prod. with lap_reg for trace calc. 
                    lap_reg = torch.sum(_eye*lap_reg).pow(2)  #Square the sum. 

                    ############## KD losses #############
                    # hint and imitation loss
                    pred_source_static = torch.tensor(preds_source_model.squeeze().data, requires_grad=False)
                    source_error = F.mse_loss(pred_source_static.squeeze(),batchY.squeeze(),reduction='none')
                    phi = (1 - source_error/ nu)
                    hint_loss = torch.mean(phi*torch.norm(hint_source_emb-hint_target_emb,dim=1))
                    imitation_loss = torch.mean(phi * F.mse_loss(pred_source_static, preds_target_model.squeeze(), reduction='none')) 
                    
                    # prediction loss
                    wILI_loss = criterion(preds_target_model.squeeze(),batchY.squeeze())

                    iter_loss =  wILI_loss + _alpha*imitation_loss + _alpha*hint_loss +\
                        _lambda*region_embedding_reconstruction_loss + _beta*lap_reg + _recon_weight*iter_target_recon_loss
                    target_loss = target_loss + iter_loss
                optim.zero_grad()
                target_loss.backward(retain_graph=True)
                optim.step()
            
            total_loss += source_loss + target_loss
            total_loss_per_batch.append(total_loss.item())
            
        self.feat_module.model.eval()
        self.module_f1.eval()
        self.module_f2.eval()
        self.module_g.eval()
        self.module_h.eval()
        
        # predict in training
        preds=list()
        targets=list()
        real_inputs=list()
        for batchXReal,batchXCat,batchY,_ in data_loader_train:
            _,wILI_embedding,_,_ = self.feat_module.predict(batchXReal,batchXCat)
            out = self.module_h.forward(wILI_embedding)
            out = self.module_f1.forward(out)
            preds_target_model = self.module_f2.forward(out)
            preds.extend(preds_target_model.cpu().data.numpy().ravel().tolist())
            targets.extend(batchY.cpu().data.numpy().ravel().tolist())
            real_inputs.append(batchXReal.cpu().data.numpy())
        
        actual_values = targets
        real_inputs = np.concatenate(real_inputs)
        return preds,actual_values,real_inputs