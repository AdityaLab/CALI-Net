import numpy as np
import pandas as pd
import json
import pdb
import datetime
import networkx as nx
from sklearn.preprocessing import PowerTransformer,StandardScaler,OneHotEncoder,MinMaxScaler

class EndogenousDataset:
    def __init__(self,column_metadata_file,datainput_file,max_hist,k_week_ahead,training_end_date,
                input_feature_prefix="input_feature",num_test_weeks=3,region_shuffle=True, DEBUG=False):
        """
            @param column_metadata: File path to json file containing column metadata.
            @param datainput_file: File path to csv file containing all data.
            @param max_hist: Integer indicating the number of historical values per time series to pass as input per prediction.
            @param k_week_ahead: Integer indicating the week ahead forecast. Ex: If k_week_ahead = 1, we will train the model to perform 1 week ahead forecasts.
            @param training_end_date: The date in "%Y-%m-%d" format specifying the last date for which data is to be used for training.
 
            @param num_test_weeks: We assign the last k-weeks to be the test set i.e last k rows of the dataframe.
            @param region_shuffle: Shuffle the assembled dataset first by dateTime and then by region. This helps achieve a balanced dataset per training batch containing all regions.
        """
         
        self.data_input_file=datainput_file
        self.col_metadata_file = column_metadata_file 
        self.region_shuffle = region_shuffle 
        self.max_hist = max_hist
        self.training_end_date=datetime.datetime.strptime(training_end_date,"%Y-%m-%d") #This format should remain constant.
        self.k_week_ahead = k_week_ahead
        self.input_feature_prefix = input_feature_prefix  #Useful incase we want to use endogenous inputs.
        self.num_test_weeks=num_test_weeks
        self.DEBUG=DEBUG

        #Read Columns Metadata and create attributes.
        self.read_metadata()

        #Read Exogenous Data
        self.data = self.read_all_exogenous_data()
        
        self.num_regions = self.data[self.region_col].unique().shape[0]  #Store this for use later on.

        #Empty dict object to be updated when the data is normalized. 
        #Dict of dicts, outer dict keyed by regions, inner dict keyed by columns.
        self.scalers={region:dict() for region in self.data[self.region_col].unique()} 
        
        #Create Upshifted columns for features and create k_week_ahead target column.
        #Self.data is automatically updated to be the time series dataset.
        self.create_time_series_dataset()
        
        #Add Region One-Hot Encoding.
        self.create_categorical_features()
        
        #Call Split Scale and Assemble Features
        self.split_scale_and_assemble_features()

    #Different 
    def read_metadata(self):
        with open(self.col_metadata_file) as f:
            obj = json.load(f)
        
        self.region_col = obj['region_names_column']
        self.feature_cols = obj['feature_columns']
        self.target_col = obj['target_column']
        self.ORIGINAL_TARGET_COLUMN = obj['target_column']
        self.date_col = obj['date_column']
        self.date_format=obj['date_format']
        self.drop_cols = obj['drop_columns']
        self.normalization = obj['normalization']
        self.categorical_region_embedding_col = obj['categorical_region_embeddings_column']
    #Different    
    def read_all_exogenous_data(self):
        all_exog_data = pd.read_csv(self.data_input_file)
 
        #Drop Unwanted Columns.
        all_exog_data.drop(self.drop_cols,axis=1,inplace=True)

        #Format-DateTime Column.
        all_exog_data[self.date_col] = all_exog_data[self.date_col].apply(lambda x: datetime.datetime.strptime(x,self.date_format))

        #Drop all rows where target column doesn't have a value.
        all_exog_data = all_exog_data[pd.notnull(all_exog_data[self.target_col])]
        all_exog_data.reset_index(inplace=True,drop=True)

        return all_exog_data

    #Different 
    def create_time_series_dataset(self):
        for key,val in self.data.groupby(self.region_col):  #Group by region.
            for shift_idx in range(self.max_hist):  #Upshift max_hist times.
                for col in self.feature_cols:  #Create new upshifted feature per `feature_column`.
                    new_col='{}-{}-{}'.format(self.input_feature_prefix,col,shift_idx)
                    tmp = val[col].shift(-shift_idx)
                    self.data.loc[tmp.index,new_col] = tmp
            
            
            tmp = val[self.target_col].shift(-(self.max_hist+(self.k_week_ahead-1)))
            tmp2 = val[self.date_col].shift(-(self.max_hist+(self.k_week_ahead-1)))

            new_col = '{}_target_{}_Weeks_Ahead'.format(self.target_col,self.k_week_ahead)
            
            new_date_col = '{}_tgt_week_dateTime'.format(self.target_col)
            
            self.data.loc[tmp.index,new_col] = tmp
            self.data.loc[tmp2.index,new_date_col] = tmp2
            
        #Drop Original Target Column and Original DateTime
        self.data.drop([self.target_col,self.date_col],axis=1,inplace=True)   #Drop Original Target Column.
        self.target_col = new_col
        self.date_col = new_date_col
        
        #Drop NaN Values Once Timeseries is created i.e drop all rows where at least a single value is NaN.
        self.data.dropna(how='any',axis=0,inplace=True)
        self.data.reset_index(inplace=True)
 
    def create_categorical_features(self):
        """
            @param df: The dataframe which contains the column `region` which is to be converted into one-hot encoded feature vector.
        """
        onehot=OneHotEncoder()
        categories=self.data[self.region_col].ravel().tolist()
        onehot_vecs = onehot.fit_transform(np.array(categories)[:,None]).todense().tolist()
        self.region_onehot = {_region:one_hot for _region,one_hot in zip(categories,onehot_vecs)}
        self.data[self.categorical_region_embedding_col] = onehot_vecs
    
    def get_transformation_type(self,col):
        if self.input_feature_prefix in col:
            col_name = col.split('-')[1]
        else:
            col_name = self.ORIGINAL_TARGET_COLUMN #Target column.
        
        return self.normalization[col_name]
    
    def _fit_transform_col(self,region,col,
                    transformation='standard-scaler',NONZERO_CONST=1e-1):
        """
            Currently transformation supports standard-scaler and PowerTransformer.
        
        """

        #The following two lines are a two step process (can be combined into a single step) deliberately to avoid a UserWarning in Pandas.
        #For more info check: (https://stackoverflow.com/questions/41710789/boolean-series-key-will-be-reindexed-to-match-dataframe-index)
        per_region_data_train = self.data.loc[self.train_indices]
        per_region_data_train = per_region_data_train[per_region_data_train[self.region_col]==region][col]
        
        if transformation=='power-box-cox':
            #Set all zero values in `transformation_columns` to be NONZERO_CONSTANT*[0,1] to avoid problems with `box-cox` transformation.
            rows=np.where(per_region_data_train==0)[0]
            update_array = per_region_data_train.values
            arr = np.random.random_sample(len(rows))
            update_array[rows] = arr*NONZERO_CONST
            per_region_data_train.loc[:] =  update_array

            #Initialize Scaler
            _scaler = PowerTransformer(method='box-cox',standardize=False)

        elif transformation=="log-standard-scaler":

            rows=np.where(per_region_data_train==0)[0]
            update_array = per_region_data_train.values
            arr = np.ones(len(rows))  #Set zeroes to ones as we are going to take a natural log.
            update_array[rows] = arr
            update_array = np.log(update_array)
            per_region_data_train.loc[:] = update_array

            #Initialize Scaler
            _scaler = StandardScaler()


        elif transformation=='standard-scaler':
            _scaler = StandardScaler()       

        elif transformation=='minmax-scaler':
            _scaler = MinMaxScaler()            

        per_region_data_train.loc[:] = _scaler.fit_transform(per_region_data_train.values[:,None]).ravel()
        self.scalers[region][col] = _scaler
        
        #Update Data post fit_transformation.
        self.data.loc[per_region_data_train.index,col] = per_region_data_train
            
    def _transform_col(self,region,col,transformation='standard-scaler',NONZERO_CONST=1e-1):
        
        #The following two lines are a two step process (can be combined into a single step) deliberately to avoid a UserWarning from Pandas.
        #For more info check: (https://stackoverflow.com/questions/41710789/boolean-series-key-will-be-reindexed-to-match-dataframe-index)

        per_region_data = self.data.loc[self.test_indices]
        per_region_data = per_region_data[per_region_data[self.region_col]==region][col]
        
        if transformation=='power-box-cox':
            rows=np.where(per_region_data==0)[0]
            update_array = per_region_data.values        
            arr = np.random.random_sample(len(rows))
            update_array[rows] = arr*NONZERO_CONST
            per_region_data.loc[:] =  update_array
        
        elif transformation=="log-standard-scaler":
            #If Log-Standard Scaler, Test Data Needs To Be Transformed Into Log values before application of the scaler. The log() transformation is performed in the `elif` statement.
            rows=np.where(per_region_data==0)[0]
            update_array = per_region_data.values
            arr = np.ones(len(rows))  #Set zeroes to ones as we are going to take a natural log.
            update_array[rows] = arr
            update_array = np.log(update_array)
            per_region_data.loc[:] = update_array 
     
        _scaler = self.scalers[region][col]
        per_region_data.loc[:] = _scaler.transform(per_region_data.values[:,None]).ravel()
        
        #Update Data post transformation.
        self.data.loc[per_region_data.index,col] = per_region_data
        
    def transform(self,region,transform_cols,training=False):
        """
            This function, normalizes training and testing data.
            
            
        """    
        if self.target_col not in transform_cols: 
            transform_cols.append(self.target_col)
        
        for col in transform_cols:
            transformation = self.get_transformation_type(col)

            if training:
                #Fit Transform
                self._fit_transform_col(region,col,transformation=transformation)
                
            else:
                #Transform
                if col==self.target_col: #We don't transform the test target column as we never need it transformed.
                    continue
                self._transform_col(region,col,transformation=transformation)

    def inverse_transform(self,predictions,targets,regions,col,transform_targets=False):
        """
            This function, inverse transforms the predictions and targets.
            @param predictions: List of predictions of wILI values.
            @param targets: List of targets of wILI values.
            @param regions: The region per prediction and target.
            @param col: The column name to be inverse transformed.
            @param transform_targets: A boolean flag indicating whether or not to inverse transform target values.

        """
        tmp = pd.DataFrame(np.concatenate([np.array(predictions)[:,None],np.array(targets)[:,None],np.array(regions)[:,None]],axis=1),columns=['predictions','targets',self.region_col])
        tmp['inv_transformed_predictions'] = tmp['predictions'].values.ravel()
        for key,val in tmp.groupby(self.region_col):
            _scaler = self.scalers[key][col]
            _preds = _scaler.inverse_transform(val['predictions'].astype(float).values[:,None])
            tmp.loc[val.index,'inv_transformed_predictions'] = _preds.ravel()
         
        if transform_targets:
            tmp['inv_transformed_targets'] = tmp['targets'].values.ravel()
            for key,val in tmp.groupby(self.region_col):
                _scaler = self.scalers[key][col]
                _tgts = _scaler.inverse_transform(val['targets'].astype(float).values[:,None])
                tmp.loc[val.index,'inv_transformed_targets'] = _tgts.ravel()

        return tmp


    def train_test_split_by_date(self):
            train_indices=list()
            test_indices=list()
            for key,val in self.data.groupby(self.region_col):
                train_indices.extend(val[val[self.date_col]<=self.training_end_date].index.values.ravel().tolist())
                test_indices.append(val[val[self.date_col]>self.training_end_date].index.values.ravel().tolist()[0]) # The [0] index at the end of the test index set indicates that we are only going to test on the very next test instance (could be 1 week ahead, 2 week ahead etc.)
    
            self.train_indices=train_indices
            self.test_indices=test_indices

    def train_test_split(self):
        """
            Method to split data into training and testing sets.
            We don't actually split the data, we only assign the training and testing indices based on the 
            pandas dataframe.
            
        """
        train_indices=list()
        test_indices=list()
        for key,val in self.data.groupby(self.region_col):
            # train_indices.extend(val.iloc[:-self.num_test_weeks].index.values.ravel().tolist())
            # test_indices.extend(val.iloc[-self.num_test_weeks:].index.values.ravel().tolist())
            train_indices.extend(val[val[self.date_col]<=self.training_end_date].index.values.ravel().tolist())
            test_indices.append(val[val[self.date_col]>=self.training_end_date+datetime.timedelta(weeks=self.k_week_ahead)].index.values.ravel().tolist()[0]) # The [0] index at the end of the test index set indicates that we are only going to test on the very next test instance (could be 1 week ahead, 2 week ahead etc.)
            
        self.train_indices = train_indices
        self.test_indices = test_indices
        
    def scale(self):
        """
            Transform each feature in dataframe according to specified transformation.
            Assumption is that before scale() is calle,d the _split() function has already been called earlier in the pipeline
            to split the data into training and testing sets.
        """
        
        if not hasattr(self,'train_indices'): #Check whether data has already been split.
            #Get Training and Testing Indices if they don't exist already.
            #self.train_test_split()
            self.train_test_split_by_date()
        
        feat_cols = self.get_feature_columns()
        
        ###############   Scale Training Data (fit_transform)  ##########
        for key,val in self.data.iloc[self.train_indices].groupby(self.region_col):
            self.transform(region=key,transform_cols=feat_cols,training=True)
        
        ############  Scale Test Data (transform) ########
        for key,val in self.data.iloc[self.test_indices].groupby(self.region_col):
            self.transform(region=key,transform_cols=feat_cols,training=False)
    
    def _shuffle_by_datetime_region(self,feat,feat_cols):
        """
            Helper function which given a dataframe, shuffles by dateTime and then by region. 
            This will be helpful in creating balanced datasets per batch in the Global, Local models. Especially relevant in the Laplacian constraint model.
            @param feat: The dataframe representing the feature data containing all columns. We will return data wherein the features contain only the `feature columns` and test contains only `target_col`.
            @param feat_cols: The list of columns corresponding to the input features.
            @return (Real_Features, Categorical_Features,Target_wILI,Prediction_Dates,Prediction_Regions)
        """
        tmp = feat.copy(deep=True)
        tmp.sort_values([self.date_col,self.region_col],inplace=True)
        
        return tmp[feat_cols],tmp[self.categorical_region_embedding_col],tmp[self.target_col],tmp[self.date_col],tmp[self.region_col]

    def get_experiment_data(self):
        """
            A helper function which returns 
            @return trainX: Training Features
            @return trainY: Training Targets
            @return testX: Testing Features
            @return testY: Testing Targets
        """
        
        feat_cols = self.get_feature_columns()
        
        self.trainX = self.data.loc[self.train_indices,feat_cols]
        self.trainXCat = self.data.loc[self.train_indices,self.categorical_region_embedding_col]
        self.testX = self.data.loc[self.test_indices,feat_cols]
        self.testXCat = self.data.loc[self.test_indices,self.categorical_region_embedding_col]
        self.trainY = self.data.loc[self.train_indices,self.target_col]
        self.testY = self.data.loc[self.test_indices,self.target_col]
        if self.region_shuffle:
            #If the `region_shuffle` flag has been set to true, we will shuffle values such that the dataframe will be sorted by dateTime and then by region. This will allow
            feat = self.data.loc[self.train_indices,:] 
            self.trainX,self.trainXCat,self.trainY, self.trainDates,self.trainRegions = self._shuffle_by_datetime_region(feat,feat_cols)
            feat = self.data.loc[self.test_indices,:]
            self.testX,self.testXCat,self.testY,self.testDates,self.testRegions = self._shuffle_by_datetime_region(feat,feat_cols)
            
    def get_feature_columns(self):
        cols=list()
        for col in self.feature_cols:
            cols.extend(list(filter(lambda x: (col in x) and 
            (self.input_feature_prefix in x),self.data.columns)))

        return cols
    
    def split_scale_and_assemble_features(self):
        
        #Split into training and testing sets.
        self.train_test_split()
        
        #Scale training data by fit_transform() of training data and transform() of testing data 
        #where we create a custom scaler per region per input column.
        if not self.DEBUG:
            self.scale()

        #Retrieve the training and testing scaled feature datasets (stored as self.trainX, self.trainY,self.testX,self.testY)
        self.get_experiment_data()
        
