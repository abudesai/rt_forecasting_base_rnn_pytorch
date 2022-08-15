import sys
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin




class ReshaperToThreeD(BaseEstimator, TransformerMixin):
    '''
    Takes a dataframe of merged history and special events and returns 3d tensor of shape N x T x D 
    N: number of series which in this case is unique combinations of locations and items
    T: number of epochs (time-steps)
    D: dimensionality of data. D is 1 if no special days are provided, otherwise
    '''
    def __init__(self, id_columns, time_column, value_columns, exog_col_prefix) -> None:
        super().__init__()
        if not isinstance(id_columns, list):
            self.id_columns = [id_columns]
        else:
            self.id_columns = id_columns

        self.time_column = time_column

        if not isinstance(value_columns, list):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns

        self.exog_col_prefix = exog_col_prefix
        self.exog_cols = None
        self.id_vals = None
        self.time_periods = None

    
    def fit(self, X, y=None):   
        self.id_vals = X[self.id_columns].drop_duplicates()
        self.time_periods = sorted(X[self.time_column].unique())
        self.exog_cols = [ c for c in X.columns if self.exog_col_prefix in c ]
        return self 


    def transform(self, X):     
        N = self.id_vals.shape[0]      
        D = len(self.value_columns + self.exog_cols)
        X = X[self.value_columns + self.exog_cols].values.reshape( (N, -1, D) )        
        # sys.exit()
        return X
        


    def inverse_transform(self, preds):  
        epoch_cols = self.time_periods[-preds.shape[1]:]
        epoch_numeric = [i for i in range(len(epoch_cols))]
        
        df1 = pd.DataFrame(self.id_vals.values, columns=self.id_columns)
        df2 = pd.DataFrame(preds, columns=epoch_numeric)
        
        preds_df = pd.concat([df1,df2], axis=1, ignore_index=True)
        preds_df.columns = self.id_columns + epoch_numeric
                
        # unpivot given dataframe
        preds_df = pd.melt(preds_df, 
            id_vars=self.id_columns,
            value_vars=epoch_numeric,
            var_name = self.time_column,
            value_name = 'prediction'
            )
        epoch_mapping = {i:t for i,t in zip(epoch_numeric, epoch_cols)}
        preds_df[self.time_column] = preds_df[self.time_column].map(epoch_mapping) 
        return  preds_df
    
    
class SubTimeSeriesSampler(BaseEstimator, TransformerMixin):
    ''' Samples a sub-series of length t <= the original series of length T. Assumes series is in columns 
    Original time-series time labels (column headers) are replaced with t_0, t_1, ... t_<series_len>.
    '''
    def __init__(self, series_len, num_reps): 
        self.series_len = series_len
        self.num_reps = num_reps


    def fit(self, X, y=None): return self


    def transform(self, X):
        curr_len = X.shape[1]
        if curr_len < self.series_len: 
            raise Exception("History isnt long enought to sample series {self.num_reps} times. Try smaller history length multiplier.")
        elif curr_len == self.series_len: 
            # cannot subsample because required len == available length
            return X 
        else:
            sampled_data = []
            # for _ in range(self.num_reps):
            #     for i in range(X.shape[0]):
            #         rand_idx = np.random.randint(0, curr_len - self.series_len)
            #         sampled_data.append( np.expand_dims( X[i, rand_idx: rand_idx + self.series_len, :] , axis=0 ) )        
            
            N, T, _ = X.shape
            num_samples = T - self.series_len + 1            
            possible_samples = N * num_samples
            max_samples = 25000
            sampling_rate = min(1.0, max_samples / possible_samples)
            for i in range(N):
                for start_idx in range(num_samples):
                    if np.random.rand() > sampling_rate: continue
                    sample = np.expand_dims( X[i, start_idx: start_idx + self.series_len, :] , axis=0)
                    sampled_data.append(sample)
            sampled_data = np.concatenate(sampled_data, axis=0)
        return sampled_data
    
        
class AddLeftRightFlipper(BaseEstimator, TransformerMixin):
    '''
    Adds left right flipped version of tensor
    '''
    def __init__(self, axis_to_flip): 
        self.axis_to_flip = axis_to_flip 

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = np.concatenate([np.flip(X, axis=self.axis_to_flip), X ], axis=0)
        return X
    
    
class SeriesLengthTrimmer(BaseEstimator, TransformerMixin):
    '''
    Trims the length of a series to use latest data points 
    '''
    def __init__(self, target_len): 
        self.target_len = target_len

    def fit(self, X, y=None): return self

    def transform(self, X):
        curr_len = X.shape[1]

        if curr_len < self.target_len: 
            raise Exception(f"Error trimming series. Given length = {curr_len} is lower than target = {self.target_len}")
        else:
            X = X[:, -self.target_len:, :]
            
        return X
    

class TSMinMaxScaler(BaseEstimator, TransformerMixin):
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, encode_len, upper_bound = 3.5):  
        self.encode_len = encode_len
        self.upper_bound = upper_bound
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        

    def fit(self, X, y=None):          
        if X.shape[1] < self.encode_len: 
            raise Exception(''' Error. Given series is shorter. 
                    Length T must equal to encode_len for train task, 
                    or (encode_len + decode_len) for prediction task. 
                    Neither is true. 
                            ''')            

        if self.encode_len < 1: 
            msg = f''' Error scaling series. 
            encode_len needs to be at least 2. Given length is {self.encode_len}.  '''
            raise Exception(msg)
        
        self.min_vals_per_d = np.expand_dims( X[ :,  : self.encode_len , : ].min(axis=1), axis = 1)
        self.max_vals_per_d = np.expand_dims( X[ :,  : self.encode_len , : ].max(axis=1), axis = 1)
        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d

        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)  
        return self

    
    def transform(self, X, y=None):  
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )  
        X = np.where( X < self.upper_bound, X, self.upper_bound)  
        X = np.where( X < -self.upper_bound, -self.upper_bound, X)  
        return X
        

    def inverse_transform(self, X): 
        assert X.shape[0] == self.min_vals_per_d.shape[0], "Error: Dimension of array to scale doesn't match fitted array."
        X = X * self.range_per_d[:, :, 0] 
        X = X + self.min_vals_per_d[:, :, 0]
        return X
    

 
class TimeSeriesXYSplitter(BaseEstimator, TransformerMixin):
    '''Splits the time series into X (history) and Y (forecast) series'''
    def __init__(self, encode_len, decode_len):
        self.encode_len = encode_len
        self.decode_len = decode_len
        

    def fit(self, X, y=None): return self

    def transform(self, X, y=None): 
        T = X.shape[1]
        if T != self.encode_len and T != self.encode_len + self.decode_len:  
            raise Exception(''' Error. Cannot split into X and Y. 
                    Length T must equal to encode_len for train task, 
                    or (encode_len + decode_len) for prediction task. 
                    Neither is true. 
                            ''')
            
        if T == self.encode_len: 
            return {
                'X': X, 
                'y': None
                }     
        else:                
            # extract the X values
            # X_adj = X[ :, -( self.encode_len + self.decode_len) : - self.decode_len , :].copy()                 
            
            # # extract the X values
            X_adj = X[ :, -( self.encode_len + self.decode_len) :  , :].copy()     
            # make all values in forecasted horizon -1 so model can recognize it as different       
            X_adj[ :, -self.decode_len :, :1 ] = -1   
            
            is_backcast = np.ones(shape=(X_adj.shape[0], X_adj.shape[1], 1))
            is_backcast[:, -self.decode_len:, :] = 0
            X_adj = np.append(X_adj, is_backcast, axis=2)
            
                 
            # y = X[ :,  -self.decode_len : , 0]
            # y_missing = X[ :,  -self.decode_len : , 1]
            y = X[ :,  -( self.encode_len + self.decode_len) : , 0]
            y_missing = X[ :,  -( self.encode_len + self.decode_len) : , 1]
            
            y_missing_sum = y_missing.max(axis=1)
            idx = y_missing_sum == 0
            y_missing[idx] = 1
            
            # print(X_adj.shape, y.shape, y_missing.shape)
            # sys.exit()
            return {
                'X': X_adj, 
                'y': y,
                'y_missing': y_missing,
                }