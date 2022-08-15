
from datetime import timedelta
import joblib
import os
import sys
import pprint
import numpy as np, pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import algorithm.utils as utils
import algorithm.model.forecaster as forecaster



# get model configuration parameters 
model_cfg = utils.get_model_config()
pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
  

def load_inference_model(model_artifacts_path, pp_params):
    model = InferenceModel(model_artifacts_path, pp_params)
    return model



class InferenceModel:
    '''
    This is the inference model used for predictions. 
    
    '''
    def __init__(self, model_artifacts_path, pp_params) -> None:    
        '''
        Load the model's preprocessor pipeline and the model. 
        '''    
        self.pp_params = pp_params
        self.pred_pipeline = load_pred_pipeline(model_artifacts_path)
        self.model = load_model(model_artifacts_path)
        self.set_training_attributes()
        self.train_data = load_train_data(model_artifacts_path, epoch_field = self.epoch_field)    
        
        self.train_data[self.id_cols]  = self.train_data[self.id_cols].astype(str)  
    
    
    
    def set_training_attributes(self):
        # forecast length that the forecaster was trained to produce
        self.decode_len = self.pred_pipeline[pp_step_names['XY_SPLITTER']].decode_len
        # names of id columns used in training
        self.id_cols = self.pred_pipeline[pp_step_names['RESHAPER_TO_THREE_D']].id_columns
        # history epochs used in training. we need the last epoch, and the timedelta
        self.fitted_epochs = self.pred_pipeline[pp_step_names['RESHAPER_TO_THREE_D']].time_periods
        # epoch_field
        self.epoch_field = self.pp_params['epoch_field']
    
     
     
    def predict(self, requested_fcst_horizon):
        '''
        Perform predictions in two steps: 
        1) transform the data as per model's needs. 
        2) make predictions using the forecast algorithm (nbeats)
        '''    
            
        # get start date/time of requested forecast
        first_epoch_in_fcst = requested_fcst_horizon[self.epoch_field].drop_duplicates().min()
        
        # since our forecaster is trained on a specific window after history ends, 
        # let's check what's the forecast window is available to us        
        available_epochs_df = self._get_forecast_horizon_epochs(first_epoch_in_fcst)    
        # print(available_epochs_df) ;  sys.exit() 
        
        # get locations and items in requested forecast
        fcst_locs_and_items_df = requested_fcst_horizon[self.id_cols].drop_duplicates()      
         
        # create the forecast df of available locations, items, and epochs  
        available_fcst_horizon = self.make_future_dataframe(fcst_locs_and_items_df, available_epochs_df)
        # print(requested_fcst_horizon.shape, available_fcst_horizon.shape)
        
        # check if requested forecast is valid (i.e. does it overlap with available forecast horizon) 
        # throw an exception otherwise       
        self.check_if_valid_forecast_horizon(requested_fcst_horizon, available_fcst_horizon)
                
        # prepare the forecast horizon data
        '''
        We do the following because the requested forecast horizon might be longer than
        what the model is trained to forecast. This is a user error, but it's not fatal. 
        We just want to trim it to be the period we can forecast for. 
        '''
        join_cols= self.id_cols + [self.epoch_field]
        final_fcst_horizon = available_fcst_horizon.merge(requested_fcst_horizon,
                                                              on = join_cols, how='left')
        # we need to fill any null values with zero. 
        # this will only happen if the user sent a forecast horizon which doesnt align
        # with what the forecaster can accomodate. 
        value_cols = [col for col in final_fcst_horizon.columns if col not in join_cols]
        final_fcst_horizon[value_cols] = final_fcst_horizon[value_cols].fillna(0.)     
                  
        
        # get training data (i.e. history) - we need it for forecasting again
        filtered_train_data = self._get_train_data(fcst_locs_and_items_df, first_epoch_in_fcst) 
        
        # remove lcoations and items with no available history 
        final_fcst_horizon = self.remove_locs_and_items_with_no_history(final_fcst_horizon, filtered_train_data)
        
        
        # concat history and forecast horizons
        final_forecast_input = pd.concat([filtered_train_data, final_fcst_horizon], axis=0, ignore_index=True)        
        final_forecast_input.sort_values( 
                    by = self.id_cols + [self.epoch_field], inplace=True)                
                
        fcst_inputs = self.pred_pipeline.fit_transform(final_forecast_input)  
        
        preds_arr = self.model.predict(fcst_inputs)
        
        # transform numpy array of preds to dataframe 
        epochs_list = available_epochs_df[self.epoch_field].tolist()
        preds_df = self._transform_preds_as_df(preds_arr, epochs_list)
        return preds_df
    
    
    
    def remove_locs_and_items_with_no_history(self, final_fcst_horizon, filtered_train_data): 
        avail_locs_and_items_df = filtered_train_data[self.id_cols].drop_duplicates()  
        final_fcst_horizon = final_fcst_horizon.merge(avail_locs_and_items_df, on = self.id_cols)
        return final_fcst_horizon
    
    
    def _transform_preds_as_df(self, preds_arr, epochs_list):
        # rescale data 
        preds_arr = self.pred_pipeline[pp_step_names['MINMAX_SCALER']].inverse_transform(preds_arr)
        preds_arr = np.squeeze(preds_arr)
        if len(preds_arr.shape) == 1: preds_arr = preds_arr.reshape((1, -1))
        
        preds_df = self.pred_pipeline[pp_step_names['RESHAPER_TO_THREE_D']].inverse_transform(preds_arr)
        
        return preds_df
        


    def check_if_valid_forecast_horizon(self, requested_fcst_horizon, available_fcst_horizon):
        merged = requested_fcst_horizon.merge(available_fcst_horizon, 
                                              on=self.id_cols + [self.epoch_field])
        
        # verify that requested (given) forecast horizon is valid.
        if merged.empty:
            fcst_start_date = requested_fcst_horizon[self.epoch_field].min()
            fcst_end_date = requested_fcst_horizon[self.epoch_field].max()
            first_avai_date = available_fcst_horizon[self.epoch_field].min()
            last_avai_date = available_fcst_horizon[self.epoch_field].max()
            msg = f'''
                This Forecaster is trained on a fixed forecast window. 
                You requested forecast from {fcst_start_date} to {fcst_end_date}. 
                Based on available history, the forecast horizon can be {first_avai_date} to {last_avai_date}. 
                Provide more history and retrain, or re-train with longer forecast window 
                if you want the forecast to extend beyond available horizon.             
            '''            
            raise Exception(msg)        
        return 


    def _get_train_data(self, fcst_locs_and_items_df, first_epoch_in_fcst):   
        idx = self.train_data[self.epoch_field] < first_epoch_in_fcst 
        filtered_train_data = self.train_data.loc[idx].copy()          
        filtered_train_data = filtered_train_data.merge(fcst_locs_and_items_df, on=self.id_cols).copy()
        filtered_train_data.sort_values(by=self.id_cols + [self.epoch_field])
        return filtered_train_data



    def make_future_dataframe(self, fcst_locs_and_items_df, forecast_epochs_df): 
        future_df = fcst_locs_and_items_df.assign(foo=1).merge(forecast_epochs_df.assign(foo=1)).drop('foo', 1)          
        return future_df
    
    
    def _get_forecast_horizon_epochs(self, first_epoch_in_fcst):        
        fitted_epochs_before_fcst_start = [ e for e in self.fitted_epochs if e < first_epoch_in_fcst]
        last_history_epoch = pd.Timestamp(max(fitted_epochs_before_fcst_start))   
        
        forecast_granularity = self.pp_params['forecast_granularity'].lower()
        if forecast_granularity == 'hourly':
            delta = timedelta(hours=1)
        elif forecast_granularity == 'daily':
            delta = timedelta(days=1)
        elif forecast_granularity == 'weekly':
            delta = timedelta(days=7)
        elif forecast_granularity == 'monthly':
            delta = relativedelta(months=1)
        elif forecast_granularity == 'yearly':
            delta = relativedelta(years=1)
        else: 
            raise ValueError(f"Unrecognized forecast granularity: {forecast_granularity}")
        
        if forecast_granularity in ['hourly', 'daily', 'weekly']:
            epochs = [ last_history_epoch + (step + 1) * delta for step in range(self.decode_len) ]
        elif forecast_granularity in ['monthly', 'yearly']: 
            dt = date(last_history_epoch.year, last_history_epoch.month, last_history_epoch.day)
            epoch_dates = [ dt + (step + 1) * delta for step in range(self.decode_len) ]
            epochs = [datetime.combine(epoch_date, datetime.min.time()) for epoch_date in epoch_dates]
        
        epochs = [pd.to_datetime(e) for e in epochs]
        # cast to 2-d numpy array
        epochs = np.array(epochs).reshape((-1,1))
        forecast_epochs_df = pd.DataFrame(epochs, columns=[self.epoch_field])  
        return forecast_epochs_df


def load_pred_pipeline(model_path):
    model_pred_pipeline = joblib.load(os.path.join(model_path, forecaster.model_pred_pipeline_fname))  
    return model_pred_pipeline


def load_model(model_path):     
    model = forecaster.Forecaster.load(model_path)
    return model


def load_train_data(model_path, epoch_field):     
    train_data = pd.read_csv(os.path.join(model_path, forecaster.train_data_fname_zip), parse_dates=[epoch_field])
    return train_data