import sys, os
import joblib

import algorithm.preprocessing.pipelines as pipelines
from algorithm.utils import get_model_config


model_cfg = get_model_config()
preprocessor_fname = "data_preprocessor.save"

class DataPreprocessor:
    def __init__(self, pp_params, has_sp_events=False):
        self.history_pipeline = pipelines.get_history_pipeline(pp_params)
        self.has_sp_events = has_sp_events
        self.sp_events_pipeline = pipelines.get_sp_events_pipeline(pp_params) if has_sp_events else None
        # These two are needed for merging history and sp_events dataframes
        self.epoch_field = pp_params['epoch_field']
        self.se_epoch_field = pp_params['se_epoch_field']
        # The prefix is used to identify sp_events columns which are created by the sp_events pipeline
        # exog refers to exogenous which just represents special events which are exogenous to the main timeseries
        self.exog_col_prefix = model_cfg['exog_col_prefix']
    
    def fit(self, history, sp_events=None):        
        self.history_pipeline.fit(history)        
        if self.sp_events_pipeline is not None:
            self.sp_events_pipeline.fit(sp_events)  
        return self
    
    def transform(self, history, sp_events=None):
        history = self.history_pipeline.transform(history) 
        if self.sp_events_pipeline is not None:
            sp_events = self.sp_events_pipeline.transform(sp_events)              
            history = history.merge(
                sp_events.head().rename(columns={self.se_epoch_field: self.epoch_field}), 
                on = [self.epoch_field],
                how = 'left'
            )
            # fill NAs
            cols = [ col for col in history.columns if self.exog_col_prefix in col]
            history[cols] = history[cols].fillna(0)
        
        return history
    
    def fit_transform(self, history, sp_events=None):
        self.fit(history, sp_events)
        history = self.transform(history, sp_events)
        return history
    

def save_data_preprocessor(data_preprocessor, file_path):
    joblib.dump(data_preprocessor, os.path.join(file_path, preprocessor_fname))  
    


def load_data_preprocessor(file_path): 
    data_preprocessor = joblib.load(os.path.join(file_path, preprocessor_fname))
    return data_preprocessor