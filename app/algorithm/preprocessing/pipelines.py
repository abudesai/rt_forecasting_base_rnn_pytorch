
import sys, os
import joblib
from sklearn.pipeline import Pipeline

import algorithm.preprocessing.preprocessors as preprocessors
from algorithm.utils import get_model_config



# get model configuration parameters 
model_cfg = get_model_config()
pp_step_names = model_cfg["pp_params"]["pp_step_names"] 


def get_history_pipeline(pp_params):     
    main_pipeline = Pipeline(
        [
            (
                pp_step_names["DUMMY_COL_CREATOR"],
                preprocessors.DummyColumnCreator(
                    col_name = pp_params['target_field'], 
                    assigned_val = 0
                    ),
            ),
            (
                pp_step_names["STRING_TYPE_CASTER"],
                preprocessors.StringTypeCaster(
                    fields = [pp_params['location_field'], pp_params['item_field']], 
                    ),
            ),
            (
                pp_step_names["MISSING_VALUE_TAGGER"],
                preprocessors.MissingValueTagger(
                    id_fields = [pp_params['location_field'], pp_params['item_field']], 
                    value_field = pp_params['target_field'], 
                    missing_field = model_cfg["missing_val_field"],
                    missing_tag = pp_params["missing_value_tag"],
                    ),
            ),
            (
                pp_step_names["FLOAT_TYPE_CASTER"],
                preprocessors.FloatTypeCaster(
                    fields = pp_params['target_field'], 
                    ),
            ),
            (
                pp_step_names["DATETIME_TYPE_CASTER"],
                preprocessors.DateTimeCaster(
                    fields = pp_params['epoch_field'], 
                    ),
            ),
            (
                pp_step_names["EPOCH_RESETTER"],
                preprocessors.EpochResetter(
                    time_column = pp_params['epoch_field'], 
                    time_granularity = pp_params['forecast_granularity'], 
                    epochBoundary = pp_params['epoch_boundary'], 
                    ),
            ),
            (
                pp_step_names["VALUE_AGGREGATOR"],
                preprocessors.ValueAggregator(
                    group_by_cols = [pp_params['location_field'], pp_params['item_field'], pp_params['epoch_field']], 
                    aggregated_columns = [pp_params['target_field'], model_cfg["missing_val_field"]],
                    ),
            ),
            (
                pp_step_names["MISSING_INTERVAL_FILLER"],
                preprocessors.MissingIntervalFiller(
                    id_columns = [pp_params['location_field'], pp_params['item_field']], 
                    time_column = pp_params['epoch_field'], 
                    value_columns = [pp_params['target_field'], model_cfg["missing_val_field"]],
                    time_granularity = pp_params['forecast_granularity']
                ),
            ),
            (
                pp_step_names["NA_FILLER"],
                preprocessors.NAFiller(
                    fields = pp_params['target_field'], 
                    fill_val = 0
                ),
            ),
        ]
    )    
    return main_pipeline
    


def get_sp_events_pipeline(pp_params):
    pipeline = Pipeline(
        [            
            (
                pp_step_names["DATETIME_TYPE_CASTER"],
                preprocessors.DateTimeCaster(
                    fields = pp_params['se_epoch_field'], 
                    ),
            ),
            (
                pp_step_names["EPOCH_RESETTER"],
                preprocessors.EpochResetter(
                    time_column = pp_params['se_epoch_field'], 
                    time_granularity = pp_params['forecast_granularity'], 
                    epochBoundary = pp_params['epoch_boundary'], 
                    ),
            ),
            (
                pp_step_names["MISSING_INTERVAL_FILLER"],
                preprocessors.MissingIntervalFiller(
                    id_columns = [], 
                    time_column = pp_params['se_epoch_field'], 
                    value_columns = [pp_params['event_field'], pp_params['window_lower'], pp_params['window_upper']], 
                    time_granularity = pp_params['forecast_granularity']
                    ),
            ),
            (
                pp_step_names["NA_FILLER"],
                preprocessors.NAFiller(
                    fields = pp_params['event_field'], 
                    fill_val = model_cfg["regular_epoch_flag"]
                    ),
            ),
            (
                pp_step_names["SP_EVENTS_NEIGH_MARKER"],
                preprocessors.SpEventsNeighborhoodMarker(
                    event_field = pp_params['event_field'], 
                    regular_epoch_lbl=model_cfg["regular_epoch_flag"],
                    epochs_before_field=pp_params['window_lower'],
                    epochs_after_field=pp_params['window_upper'],
                    epochs_before_suffix=model_cfg["se_epochs_before_suffix"],
                    epochs_after_suffix=model_cfg["se_epochs_after_suffix"]
                ),
            ),
            (
                pp_step_names["SP_EVENTS_OHE"],
                preprocessors.OneHotEncoderMultipleCols(
                    ohe_columns = pp_params['event_field'], 
                    prefix= model_cfg['exog_col_prefix'], 
                    fill_na_val = 0,
                    max_num_categories=100
                    ),
            ),          
            (
                pp_step_names["DATETIME_TYPE_CASTER2"],
                preprocessors.DateTimeCaster(
                    fields = pp_params['se_epoch_field'], 
                    ),
            ),
            (
                pp_step_names["COLUMNS_SELECTOR"],
                preprocessors.ColumnsSelector(
                    columns = [pp_params['event_field'], pp_params['window_lower'], pp_params['window_upper']], 
                    selector_type = "drop"
                    ),
            ),
        ]
    )    
    return pipeline



# def apply_preprocess_transformation(data, pipeline_obj):
#     history = data["history"] ; sp_events = data["sp_events"]  
#     history_pp_pipe = pipeline_obj["history_pp_pipe"]
#     sp_events_pp_pipe = pipeline_obj["sp_events_pp_pipe"]
    
#     history = history_pp_pipe.transform(data)
#     if sp_events is not None:    
#         sp_events = sp_events_pp_pipe.transform(sp_events)
        

# def save_preprocessor(pipeline_obj, file_path):    
#     joblib.dump(pipeline_obj["history_pp_pipe"], os.path.join(file_path, history_pp_fname))  
#     if pipeline_obj["sp_events_pp_pipe"] is not None: 
#         joblib.dump(pipeline_obj["sp_events_pp_pipe"], os.path.join(file_path, sp_events_pp_fname))  
    

# def load_preprocessor(file_path):        
#     file_path_and_name = os.path.join(file_path, history_pp_fname)
#     if not os.path.exists(file_path_and_name):
#         raise Exception(f'''Error: No trained history preprocessor found. 
#         Expected to find model files in path: {file_path_and_name}''')       
    
#     history_pp_pipe = joblib.load(file_path_and_name)  
    
#     file_path_and_name = os.path.join(file_path, sp_events_pp_fname)
#     if not os.path.exists(file_path_and_name):
#         print(f'''Error: No trained sp_events preprocessor found. ''')       
#         sp_events_pp_pipe = None
#     else:         
#         sp_events_pp_pipe = joblib.load(file_path_and_name) 
    
#     pipeline_obj = {
#         "history_pp_pipe": history_pp_pipe,
#         "sp_events_pp_pipe": sp_events_pp_pipe
#     }    
#     return pipeline_obj 
    