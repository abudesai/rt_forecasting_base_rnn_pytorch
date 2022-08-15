
from pydoc import doc
import numpy as np, pandas as pd, random
import sys, os
import glob
import json
os.environ['PYTHONHASHSEED']=str(2)
import torch as T



def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
    else: 
        print(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def read_csv_files_in_dir(data_path): 
    all_files = os.listdir(data_path) 
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))  
    input_files = [ os.path.join(data_path, file) for file in csv_files ]
    if len(input_files) == 0:
        data = None 
    else:         
        raw_data = [ pd.read_csv(file, dtype=str) for file in input_files ]
        data = pd.concat(raw_data)
    return data


def get_data(data_path):       
    history_data_path = os.path.join(data_path, "forecastingBaseHistory")
    history = read_csv_files_in_dir(history_data_path)
    if history is None: raise Exception(f'No history data found in {history_data_path}.')
    sp_events = read_csv_files_in_dir(os.path.join(data_path, "forecastingBaseSpecialEvents"))
    data = { "history": history, "sp_events": sp_events }
    return data


def get_data_schema(data_schema_path): 
    try: 
        json_files = list(filter(lambda f: f.endswith('.json'), os.listdir(data_schema_path) )) 
        if len(json_files) > 1: raise Exception(f'Multiple json files found in {data_schema_path}. Expecting only one schema file.')
        full_fpath = os.path.join(data_schema_path, json_files[0])
        with open(full_fpath, 'r') as f:
            data_schema = json.load(f)
            return data_schema  
    except: 
        raise Exception(f"Error reading data_schema file at: {data_schema_path}")  
      

def get_json_file(file_path, file_type): 
    try:
        json_data = json.load(open(file_path)) 
        return json_data
    except: 
        raise Exception(f"Error reading {file_type} file at: {file_path}")   


def get_hyperparameters(hyper_param_path): 
    hyperparameters_path = os.path.join(hyper_param_path, 'hyperparameters.json')
    return get_json_file(hyperparameters_path, "hyperparameters")


def get_model_config():
    model_cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'model_config.json')
    return get_json_file(model_cfg_path, "model config")


def get_hpt_specs():
    hpt_params_path = os.path.join(os.path.dirname(__file__), 'config', 'hpt_params.json')
    return get_json_file(hpt_params_path, "HPT config")
    

def save_json(file_path_and_name, data):
    """Save json to a path (directory + filename)"""
    with open(file_path_and_name, 'w') as f:
        json.dump( data,  f, 
                  default=lambda o: make_serializable(o), 
                  sort_keys=True, 
                  indent=4, 
                  separators=(',', ': ') 
                  )

def make_serializable(obj): 
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return json.JSONEncoder.default(None, obj)


def print_json(result):
    """Pretty-print a jsonable structure"""
    print(json.dumps(
        result,
        default=lambda o: make_serializable(o), 
        sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
    

def save_dataframe(df, save_path, file_name): 
    df.to_csv(os.path.join(save_path, file_name), index=False)


def get_train_test_split(processed_history, pp_params):
    model_cfg = get_model_config()
    
    id_fields = [pp_params['location_field'] , pp_params['item_field']]
    uniq_locs_and_items = processed_history[id_fields].drop_duplicates().reset_index(drop=True)
    
    N = uniq_locs_and_items.shape[0] 
    if N >= 100:
        valid_items = uniq_locs_and_items.sample(frac = model_cfg["valid_split"])
        train_items = uniq_locs_and_items.drop(valid_items.index)        
        
        history_train = processed_history.merge(train_items, on=id_fields)
        history_valid = processed_history.merge(valid_items, on=id_fields)
    else: 
        history_train = processed_history
        history_valid = None
            
    return history_train, history_valid