
from ctypes import util
import numpy as np
import uuid
import time
import math
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import os
import warnings
import sys
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer
import algorithm.preprocessing.preprocess_utils as pp_utils

# get model configuration parameters 
model_cfg = utils.get_model_config()



def get_hpt_space(hpt_specs): 
    param_grid = []
    for hp_obj in hpt_specs: 
        if hp_obj["run_HPO"] == False:
            param_grid.append( Categorical([hp_obj['default']], name=hp_obj['name']) )        
        elif hp_obj["type"] == 'categorical':
            param_grid.append( Categorical(hp_obj['categorical_vals'], name=hp_obj['name']) )
        elif hp_obj["type"] == 'int' and hp_obj["search_type"] == 'uniform':
            param_grid.append( Integer(hp_obj['range_low'], hp_obj['range_high'], prior='uniform', name=hp_obj['name']) )
        elif hp_obj["type"] == 'int' and hp_obj["search_type"] == 'log-uniform':
            param_grid.append( Integer(hp_obj['range_low'], hp_obj['range_high'], prior='log-uniform', name=hp_obj['name']) )
        elif hp_obj["type"] == 'real' and hp_obj["search_type"] == 'uniform':
            param_grid.append( Real(hp_obj['range_low'], hp_obj['range_high'], prior='uniform', name=hp_obj['name']) )
        elif hp_obj["type"] == 'real' and hp_obj["search_type"] == 'log-uniform':
            param_grid.append( Real(hp_obj['range_low'], hp_obj['range_high'], prior='log-uniform', name=hp_obj['name']) )        
        else: 
            raise Exception(f"Error creating Hyper-Param Grid. \
                Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                Verify hpt_params.json file.")
    return param_grid  


def get_default_hps(hpt_specs):
    default_hps = [ hp["default"] for hp in hpt_specs ]
    return default_hps


def load_best_hyperspace(results_path):
    results = [ f for f in list(sorted(os.listdir(results_path))) if 'json' in f ]
    if len(results) == 0: return None
    best_result_name = results[0]
    best_result_file_path = os.path.join(results_path, best_result_name)
    return utils.get_json_file(best_result_file_path, "best_hpt_results")


def save_best_parameters(results_path, hyper_param_path):
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace(results_path)
    if space_best_model is None:
        print("No models yet. Continuing...")
        return
    print("Best model yet:", space_best_model['model_name'])
    print("-"*60)
    # print("Best hyperspace yet:\n",  space_best_model["space"] )
    
    # Important: you must save the best parameters to /opt/ml/model/model_config/hyperparameters.json during HPO for them to persist
    utils.save_json(os.path.join(hyper_param_path, "hyperparameters.json"), space_best_model["space"])


def have_hyperparams_to_tune(hpt_specs): 
    for hp_obj in hpt_specs: 
        if hp_obj["run_HPO"] == True: return True
    return False


def clear_hp_results_dir(results_path): 
    if os.path.exists(results_path):
        for f in os.listdir(results_path):
            os.remove(os.path.join(results_path, f))
    else: 
        os.makedirs(results_path)
        

def tune_hyperparameters(data, data_schema, num_trials, hyper_param_path, hpt_results_path):  
    # read hpt_specs file 
    hpt_specs = utils.get_hpt_specs()
    # check if any hyper-parameters are specified to be tuned
    if not have_hyperparams_to_tune(hpt_specs): 
        print("No hyper-parameters to tune.")
        return
    
    print("Running HPT ...")
    
    start = time.time() 
    
    # clear previous results, if any
    clear_hp_results_dir(hpt_results_path)       
    
    # get the hpt space (grid) and default hps
    hpt_space = get_hpt_space(hpt_specs)  
    default_hps = get_default_hps(hpt_specs)  
    
    
    # get preprocessing parameters
    pp_params = pp_utils.get_preprocess_params(data_schema)     
    
    # preprocess data (includes history and special_events if any)
    processed_history, data_preprocessor = model_trainer.preprocess_data(data, pp_params) 
    
    train_data, valid_data = utils.get_train_test_split(processed_history, pp_params)
    # print(train_data.shape, valid_data.shape); sys.exit()
    
    # Scikit-optimize objective function
    @use_named_args(hpt_space)
    def objective(**hyperparameters): 
         
        # print(hyperparameters) 
                
        # set random seeds
        utils.set_seeds()
        
        
        """Build a model from this hyper parameter permutation and evaluate its performance"""
        # train model
        train_artifacts = model_trainer.train_model(train_data, pp_params, hyperparameters)        
        
        if valid_data is not None:
            valid_inputs = train_artifacts["model_pred_pipeline"].fit_transform(valid_data)
        else: 
            # not enough data to do train/test split, so just re-use train data
            valid_inputs = train_artifacts["model_pred_pipeline"].fit_transform(train_data)
        
        score = train_artifacts["model"].evaluate(valid_inputs)   # returns mse
        
        # Our optimizing metric is the model loss fn
        opt_metric = np.round(score, 5)   # returns binary_cross_entropy
        if np.isnan(opt_metric) or math.isinf(opt_metric): opt_metric = 1.0e5     # sometimes loss becomes inf, so use a large value
        # create a unique model name for the trial - we add loss into file name 
        # so we can later sort by file names, and get the best score file without reading each file   
        model_name = f"model_{str(opt_metric)}_{str(uuid.uuid4())[:5]}"
        print("trial model:", model_name)
        # create trial result dict
        result = {       
            'model_name': model_name,    
            'space': hyperparameters, 
            'loss': opt_metric, 
            # 'history': pd.DataFrame(history.history).to_json(),
        }
        # Save training results to disks with unique filenames
        utils.save_json(os.path.join(hpt_results_path, model_name + ".json"), result)
        # Save the best model parameters found so far in case the HPO job is killed
        save_best_parameters(hpt_results_path, hyper_param_path)
        return opt_metric
    
    
    n_initial_points = int(max(1, min(num_trials/3, 5)))
    n_calls = max(2, num_trials)  # gp_minimize needs at least 2 trials, or it throws an error
    gp_ = gp_minimize(
        objective, # the objective function to minimize
        hpt_space, # the hyperparameter space
        x0=default_hps, # the initial parameters to test
        acq_func='EI', # the acquisition function
        n_initial_points=n_initial_points,
        n_calls=n_calls, # the number of total evaluations of f(x), including n_initial_points
        random_state=0
    )
    
    end = time.time()
    print(f"Total HPO time: {np.round((end - start)/60.0, 2)} minutes") 



