import os, shutil
import sys
import time 
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy

sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer
import algorithm.model_server as model_server
import algorithm.model_tuner as model_tuner
import algorithm.model.forecaster as forecaster


import scoring_utils as scoring

# paths to the input/outputs from ml_vol (volume mounted to docker, but here we are mimicking it locally)
inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data")
train_data_path = os.path.join(data_path, "training")
test_data_path = os.path.join(data_path, "testing")

model_path = "./ml_vol/model/"
model_access_path = os.path.join(model_path, "model.save")
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

# local dir to place evaluation results
test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)

# change this to whereever you placed your local testing datasets
local_datapath = "./../../datasets" 

'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os, or python-version related issues, so beware.
'''

model_name= forecaster.MODEL_NAME


def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": { "forecastingBaseHistory": None, "forecastingBaseSpecialEvents": None },
                    "testing": { "forecastingBaseHistory": None, "forecastingBaseSpecialEvents": None },
                }
            },
            "model": {
                "model_config": None, 
                "artifacts": None,
            }, 
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                 
            }
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)


def copy_example_files(dataset_name):       
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json")) 
    # data schema
    shutil.copyfile(os.path.join(local_datapath, dataset_name, f"{dataset_name}_schema.json"), os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    
    # train history data    
    fname = f"{dataset_name}_history_train.csv"    
    shutil.copyfile(os.path.join(local_datapath, dataset_name, fname), 
                    os.path.join(train_data_path, "forecastingBaseHistory", fname) )
     
    # test history data    
    fname = f"{dataset_name}_history_test.csv"   
    shutil.copyfile(os.path.join(local_datapath, dataset_name, fname), 
                    os.path.join(test_data_path, "forecastingBaseHistory", fname) )    
    
    # train and test special_events data  
    fname = f"{dataset_name}_special_events.csv"    
    if os.path.exists(os.path.join(local_datapath, dataset_name, fname)) :
        shutil.copyfile(os.path.join(local_datapath, dataset_name, fname), 
                        os.path.join(train_data_path, "forecastingBaseSpecialEvents", fname) )
        shutil.copyfile(os.path.join(local_datapath, dataset_name, fname), 
                        os.path.join(test_data_path, "forecastingBaseSpecialEvents", fname) )


def run_HPT(num_hpt_trials): 
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    model_tuner.tune_hyperparameters(train_data, data_schema, num_hpt_trials, hyper_param_path, hpt_results_path)


def train_and_save_algo():        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data - this returns a dictionary with history and sp_events data 
    train_data = utils.get_data(train_data_path)  
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # get trained preprocessor, model, training history 
    model_artifacts, data_preprocessor = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)            
    # Save the training artifacts
    model_trainer.save_training_artifacts(model_artifacts, data_preprocessor, model_artifacts_path) 
    print("done with training")


def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model 
    predictor = model_server.ModelServer(model_artifacts_path, data_schema)
    # make predictions
    predictions = predictor.predict(test_data)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    results = score(test_data, predictions) 
    print("done with predictions")
    return results



def set_id_and_target_cols(dataset_name):
    global loc_col, item_col, epoch_col, target_col, missing_val_tag
    data_schema = utils.get_data_schema(data_schema_path)
    # set the location attribute
    loc_col = data_schema["inputDatasets"]["forecastingBaseHistory"]["locationField"]       
    # set the item attribute
    item_col = data_schema["inputDatasets"]["forecastingBaseHistory"]["itemField"]       
    # set the epoch attribute
    epoch_col = data_schema["inputDatasets"]["forecastingBaseHistory"]["epochField"]     
    # set the target attribute
    target_col = data_schema["inputDatasets"]["forecastingBaseHistory"]["targetField"]      
    # set the target attribute
    missing_val_tag = data_schema["datasetSpecs"]["missingValueTag"]  


def score(test_data, predictions):    
    
    test_data = test_data["history"] 
    test_data = test_data[test_data[target_col] != missing_val_tag]
    test_data[epoch_col] = pd.to_datetime(test_data[epoch_col])
    test_data[target_col] = test_data[target_col].astype(np.float32)   
    
    # remove missing values, if any 
    test_data = test_data.loc[test_data[target_col] != missing_val_tag]
    
    predictions = test_data[[loc_col, item_col, epoch_col, target_col]].merge(predictions, 
                                    on=[loc_col, item_col, epoch_col])
    
    predictions.to_csv(os.path.join(test_results_path, "actuals_and_predictions.csv"), index=False)
    
    # score the results
    scores = get_averaged_scores(predictions)    
    
    scores["perc_pred_missing"] = np.round( 100 * (1 - predictions.shape[0] / test_data.shape[0]), 2)
    return scores



def get_averaged_scores(predictions):
    id_cols = [loc_col, item_col]
    locs_and_items_df = predictions[id_cols].drop_duplicates()  
    
    scores = { "rmse": [], "mae": [], "nmae": [], "r2": [], "mape": [], "smape": [], "wape": []}
    
    predictions[target_col] = predictions[target_col].astype(np.float32)
    predictions['prediction'] = predictions['prediction'].astype(np.float32)
    
    for i, row in locs_and_items_df.iterrows(): 
        loc_id = row[loc_col]
        item_id = row[item_col]
        
        idx = (predictions[loc_col] == loc_id) & (predictions[item_col] == item_id)
        filtered = predictions.loc[idx]
        rmse = mean_squared_error(filtered[target_col], filtered['prediction'], squared=False)
        mae = mean_absolute_error(filtered[target_col], filtered['prediction'])
        
        mape = scoring.get_mape(filtered[target_col], filtered['prediction'])
        smape = scoring.get_smape(filtered[target_col], filtered['prediction'])
        wape = scoring.get_wape(filtered[target_col], filtered['prediction'])
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(filtered['prediction'], filtered[target_col])
        r2 = r_value ** 2
          
        q3, q1 = np.percentile(filtered[target_col], [75, 25])
        iqr = q3 - q1
        nmae = mae / iqr
        scores['rmse'].append(rmse)
        scores['mae'].append(mae)
        scores['nmae'].append(nmae)
        scores['mape'].append(mape)
        scores['smape'].append(smape)
        scores['wape'].append(wape)
        scores['r2'].append(r2)     
        
    scores['rmse'] = np.round(np.mean(scores['rmse']), 3)
    scores['mae'] = np.round(np.mean(scores['mae']), 3)
    scores['nmae'] = np.round(np.mean(scores['nmae']), 3)
    scores['mape'] = np.round(np.mean(scores['mape']), 3)
    scores['smape'] = np.round(np.mean(scores['smape']), 3)
    scores['wape'] = np.round(np.mean(scores['wape']), 3)
    scores['r2'] = np.round(np.mean(scores['r2']), 3)
    return scores
        


def save_test_outputs(results, run_hpt, dataset_name_split=None):    
    df = pd.DataFrame(results) if dataset_name_split is None else pd.DataFrame([results])        
    df = df[["model", "dataset_name", "run_hpt", "num_hpt_trials", 
             "rmse", "mae", "nmae", "mape", "smape", "wape", 
             "r2", "perc_pred_missing",
             "elapsed_time_in_minutes"]]    
    print(df)
    file_path_and_name = get_file_path_and_name(run_hpt, dataset_name_split)
    df.to_csv(file_path_and_name, index=False)

    

def get_file_path_and_name(run_hpt, dataset_name): 
    if dataset_name is None: 
        fname = f"_{model_name}_results_with_hpt.csv" if run_hpt else f"_{model_name}_results_no_hpt.csv"
    else: 
        fname = f"{model_name}_{dataset_name}_results_with_hpt.csv" if run_hpt else f"{model_name}_{dataset_name}_results_no_hpt.csv"
    full_path = os.path.join(test_results_path, fname)
    return full_path


def run_train_and_test(dataset_name, run_hpt, num_hpt_trials):
    start = time.time()    
    # create the directory which imitates the bind mount on container
    create_ml_vol()   
    # copy the required files for model training    
    copy_example_files(dataset_name)   
    # run HPT and save tuned hyperparameters
    if run_hpt: run_HPT(num_hpt_trials)     
    # train the model and save          
    train_and_save_algo()        
    # load the trained model and get predictions on test data
    set_id_and_target_cols(dataset_name=dataset_name)
    results = load_and_test_algo()        
    
    end = time.time()
    elapsed_time_in_minutes = np.round((end - start)/60.0, 2)
    
    results = { **results, 
               "model": model_name, 
               "dataset_name": dataset_name, 
               "run_hpt": run_hpt, 
               "num_hpt_trials": num_hpt_trials if run_hpt else None, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes 
               }
    
    print(f"Done with dataset in {elapsed_time_in_minutes} minutes.")
    return results


if __name__ == "__main__": 
    
    num_hpt_trials = 6
    
    run_hpt_list = [False, True]
    run_hpt_list = [False]
    
    splits = [1,2,3,4]
    splits = [4]
    
    datasets = ["air_quality", "airline_passengers", "beer_sales", "food_demand", 
                "stock_prices", "synthetic_daily"]
    
    datasets = ["airline_passengers"]
    
    all_results = []
    for split in splits:
        for run_hpt in run_hpt_list:
            
            for dataset_name in datasets:    
                dataset_name_split = f"{dataset_name}_{split}"
                print("-"*60)
                print(f"Running dataset {dataset_name_split} ....")
                results = run_train_and_test(dataset_name_split, run_hpt, num_hpt_trials)
                save_test_outputs(results, run_hpt, dataset_name_split)            
                all_results.append(results)
                print("-"*60)
                        
    save_test_outputs(all_results, run_hpt, dataset_name_split=None)