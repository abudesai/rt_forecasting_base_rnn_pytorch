import numpy as np, pandas as pd
import os
import sys

import algorithm.utils as utils
# import algorithm.model.forecaster as forecaster
import algorithm.model.forecaster_inference as forecaster_inference
import algorithm.preprocessing.preprocessing_main as preprocessing
import algorithm.preprocessing.preprocess_utils as pp_utils


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None 
        self.pp_params = pp_utils.get_preprocess_params(data_schema)    
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            self.preprocessor = preprocessing.load_data_preprocessor(self.model_path)
            return self.preprocessor           
        else: return self.preprocessor
    
    
    def _get_model(self): 
        if self.model is None: 
            self.model = forecaster_inference.load_inference_model(self.model_path, self.pp_params)
            return self.model
        else: return self.model
        
       
    def predict(self, data, ):    
                   
        preprocessor = self._get_preprocessor()
        model = self._get_model()   
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        # if model is None:  raise Exception("No model found. Did you train first?")
        
        # extract history and sp_events data
        history = data["history"] ; sp_events = data["sp_events"]  
        # print(history.head()); sys.exit()
        # del history["Passengers"]
        # del history["Close_Price"]
            
        # transform data - returns tuple of X (array of word indexes) and y (None in this case)
        processed_inputs = preprocessor.transform(history, sp_events) 
        # print("processed data: \n", processed_inputs.head())
        
        # del processed_inputs["__exog__missing__"]
        
        preds_df = model.predict( processed_inputs )
        
        return preds_df
