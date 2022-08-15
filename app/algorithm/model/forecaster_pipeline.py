
import sys, os
import joblib
from sklearn.pipeline import Pipeline
 

import algorithm.model.forecaster_preprocessors as preprocessors


def get_forecaster_preprocess_pipelines(pp_params, model_cfg):
    '''
    pipeline 1 steps - for both training and prediction tasks
    pivot data                        
    
    pipeline 2 steps - for training only
    sub sample                        
    left-right flipper                
    series-length trimmer (length = history_len + fcst_len)    
    
    pipeline 3 steps - for prediction only
    series-length trimmer (length = history_len)  
    
    pipeline 4 steps - for both training and prediction tasks
    min max scale                    
    x/y split                       
    '''    
    
    decode_len = int(pp_params["forecast_horizon_length"])
    encode_len = decode_len * int (pp_params["hist_len_multiple_of_fcst_len"])
    
    # pipeline 1
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    pipeline1 = Pipeline([
        (
            pp_step_names["RESHAPER_TO_THREE_D"],
            preprocessors.ReshaperToThreeD(
                id_columns = [pp_params['location_field'], pp_params['item_field']], 
                time_column = pp_params['epoch_field'], 
                value_columns = pp_params['target_field'],  
                exog_col_prefix = model_cfg['exog_col_prefix']
            ),
        )    
    ])
    
    # pipeline 2
    pipeline2 = Pipeline([
        (
            pp_step_names["SERIES_SUBSAMPLER"],
            preprocessors.SubTimeSeriesSampler(
                series_len = encode_len + decode_len,
                num_reps = model_cfg["num_subsampling_reps"]
            ),
        ),
        # do left right flip
        (
            pp_step_names["LEFT_RIGHT_FLIPPER"],
            preprocessors.AddLeftRightFlipper(axis_to_flip = 1)
        )
    ])
    
    # both
    pipeline3 = Pipeline([        
        (
            pp_step_names["SERIES_TRIMMER"],
            preprocessors.SeriesLengthTrimmer(
                target_len = encode_len + decode_len,
            )
        ),
        # Min max scale data
        (
            pp_step_names["MINMAX_SCALER"],
            preprocessors.TSMinMaxScaler(
                encode_len = encode_len,
                upper_bound = model_cfg["scaler_max_bound"]
            )
        ), 
        (
            pp_step_names["XY_SPLITTER"],
            preprocessors.TimeSeriesXYSplitter(
                encode_len = encode_len,
                decode_len = decode_len
            )
        )   
    ])
    
    training_pipeline = Pipeline( pipeline1.steps 
                                 + pipeline2.steps 
                                 + pipeline3.steps )    
    pred_pipeline = Pipeline( pipeline1.steps +  pipeline3.steps )    
    return training_pipeline, pred_pipeline

