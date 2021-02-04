import torch
import torch.nn as nn
import pdb

class GlobalRecurrentFeatureModule(nn.Module):
    """
        Model which reads in one-hot encoded real vector denoting region, a real value of case counts of past K weeks 
        of COVID cases (and possibly other exogenous datasets) and returns a single value corresponding to the 
        COVID case (or wILI case) count for the K-week ahead forecast. K can currently be equal to 1,2,3,4. 
        A new model needs to be trained for each K-week ahead forecast.
    
    """
    def __init__(self,input_size_real,num_input_features,input_size_categorical,output_size_real,output_size_categorical,
                 hidden_size_real=16,hidden_size_categorical=16,dropout_prob=0.6,num_layers=2):
        """
            @param input_size: The input feature size. 
        
        """
        super(GlobalRecurrentFeatureModule,self).__init__()
        
        self.num_layers=num_layers

        #Real Data Network
        self.recurrent_hidden_size = hidden_size_real
        self.gru = nn.GRU(num_input_features+hidden_size_categorical,hidden_size_real,num_layers=num_layers,batch_first=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size_real,affine=True) #Figure out how to apply batch-norm to GRU
        self.fc21= nn.Linear(hidden_size_real,hidden_size_real)
        self.fc22=nn.Linear(hidden_size_real,hidden_size_real)
        self.fc3 = nn.Linear(hidden_size_real,output_size_real)
 
        self.relu = nn.ReLU()       
        self.softmax = nn.Softmax(dim=1)  #Applied to output of AE
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_prob)
        #Categorical Reconstruction AE
        self.ae_in = nn.Linear(input_size_categorical,hidden_size_categorical)
        self.ae_out = nn.Linear(hidden_size_categorical,output_size_categorical)
     
        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda'

    def initHidden(self,batch_size,num_layers):
        return torch.zeros(num_layers, batch_size, self.recurrent_hidden_size)#,device=self.device) 

    def forward(self,real_input,categorical_input):
        #Categorical AE
        categorical_embedding = self.sigmoid(self.ae_in(categorical_input))
        categorical_recon = self.softmax(self.ae_out(categorical_embedding))

        #Input Batches in real_input will be of size (batch_size,sequence_size,feature_size). Here sequence_size indicates the amount of historical data being used.
        real_emb = self.initHidden(real_input.size(0),self.num_layers) 

        for i in range(real_input.size(1)):
            #During each sequence iter. of recurrence, categorical_embedding needs to be appended.
            recurrent_input = torch.cat([categorical_embedding.unsqueeze(1),real_input[:,i,:].view(-1,1,real_input.size(2))],dim=2)
            output,real_emb = self.gru(recurrent_input,real_emb)
        
        #Output embedding of the recurrent network is used by feed-forward component to predict next value.

        real_emb = self.relu(self.dropout(self.fc21(real_emb[-1,:,:])))
        real_emb = self.relu(self.dropout(self.fc22(real_emb)))
        real_out = self.fc3(real_emb)
        
        return real_out,real_emb,categorical_recon,categorical_embedding

class GlobalFeatureModule(nn.Module):
    """
        Model which reads in one-hot encoded real vector denoting region, a real value of case counts of past K weeks 
        of COVID cases (and possibly other exogenous datasets) and returns a single value corresponding to the 
        COVID case (or wILI case) count for the K-week ahead forecast. K can currently be equal to 1 or 2. 
        A new model needs to be trained for each K-week ahead forecast.
    
    """
    def __init__(self,input_size_real,input_size_categorical,output_size_real,output_size_categorical,
                 hidden_size_real=16,hidden_size_categorical=16,dropout_prob=0.6):
        """
            @param input_size: The input feature size. Curr
        
        """
        super(GlobalFeatureModule,self).__init__()
        
        #Real Data Network
        self.fc1 = nn.Linear(input_size_real+hidden_size_categorical,hidden_size_real)
        self.fc2 = nn.Linear(hidden_size_real,hidden_size_real)
        self.fc21=nn.Linear(hidden_size_real,hidden_size_real)
        self.fc22=nn.Linear(hidden_size_real,hidden_size_real)
        self.fc3 = nn.Linear(hidden_size_real,output_size_real)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size_real,affine=True)
 
        self.relu = nn.ReLU()       
        self.softmax = nn.Softmax(dim=1)  #Applied to output of AE
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_prob)
        #Categorical Reconstruction AE
        self.ae_in = nn.Linear(input_size_categorical,hidden_size_categorical)
        self.ae_out = nn.Linear(hidden_size_categorical,output_size_categorical)
        
    def forward(self,real_input,categorical_input):
        #Categorical AE
        categorical_embedding = self.sigmoid(self.ae_in(categorical_input))
        categorical_recon = self.softmax(self.ae_out(categorical_embedding))
        
        #Concatenate real_input and categorical embedding
        #Each of categorical_embedding and real_input is (batch_size , k) where k varies for categorical embedding and real_input.
        real_input = torch.cat([categorical_embedding,real_input],dim=1)
        real_emb = self.relu(self.bn1(self.fc1(real_input)))
        real_emb = self.relu(self.dropout(self.fc2(real_emb)))
        real_emb = self.relu(self.dropout(self.fc21(real_emb)))
        real_emb = self.relu(self.dropout(self.fc22(real_emb)))
        real_out = self.fc3(real_emb)
        
        return real_out,real_emb,categorical_recon,categorical_embedding
