import numpy as np, pandas as pd
import sys

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta


HOURLY_BOUNDARIES = list(np.arange(59))   # minutes
DAILY_BOUNDARIES = list(np.arange(24))    # hours
WEEKLY_BOUNDARIES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']   # weekday
MONTHLY_BOUNDARIES = list(1+np.arange(27))  # day in month
ANNUAL_BOUNDARIES = list(1+np.arange(12))   # month in year

BOUNDARIES = {
    "hourly" : HOURLY_BOUNDARIES,
    "daily" : DAILY_BOUNDARIES,
    "weekly" : WEEKLY_BOUNDARIES,
    "monthly" : MONTHLY_BOUNDARIES,
    "yearly" : ANNUAL_BOUNDARIES,
}

WKDAY_DELTAS = {
    # epochboundary, givenwkday
    ('monday', 'monday') : 0,
    ('monday', 'tuesday') : 1,
    ('monday', 'wednesday') : 2,
    ('monday', 'thursday') : 3,
    ('monday', 'friday') : 4,
    ('monday', 'saturday') : 5,
    ('monday', 'sunday') : 6,
    
    ('tuesday', 'tuesday') : 0,
    ('tuesday', 'wednesday') : 1,
    ('tuesday', 'thursday') : 2,
    ('tuesday', 'friday') : 3,
    ('tuesday', 'saturday') : 4,
    ('tuesday', 'sunday') : 5,
    ('tuesday', 'monday') : 6,
    
    ('wednesday', 'wednesday') : 0,
    ('wednesday', 'thursday') : 1,
    ('wednesday', 'friday') : 2,
    ('wednesday', 'saturday') : 3,
    ('wednesday', 'sunday') : 4,
    ('wednesday', 'monday') : 5,
    ('wednesday', 'tuesday') : 6,
    
    ('thursday', 'thursday') : 0,
    ('thursday', 'friday') : 1,
    ('thursday', 'saturday') : 2,
    ('thursday', 'sunday') : 3,
    ('thursday', 'monday') : 4,
    ('thursday', 'tuesday') : 5,
    ('thursday', 'wednesday') : 6,
    
    ('friday', 'friday') : 0,
    ('friday', 'saturday') : 1,
    ('friday', 'sunday') : 2,
    ('friday', 'monday') : 3,
    ('friday', 'tuesday') : 4,
    ('friday', 'wednesday') : 5,
    ('friday', 'thursday') : 6,
    
    ('saturday', 'saturday') : 0,
    ('saturday', 'sunday') : 1,
    ('saturday', 'monday') : 2,
    ('saturday', 'tuesday') : 3,
    ('saturday', 'wednesday') : 4,
    ('saturday', 'thursday') : 5,
    ('saturday', 'friday') : 6,
    
    ('sunday', 'sunday') : 0,
    ('sunday', 'monday') : 1,
    ('sunday', 'tuesday') : 2,
    ('sunday', 'wednesday') : 3,
    ('sunday', 'thursday') : 4,
    ('sunday', 'friday') : 5,
    ('sunday', 'saturday') : 6,
}




def get_preprocess_params(data_schema): 
    # initiate the pp_params dict
    pp_params = {}   
    
    # history data fields  
    pp_params["id_field"] = data_schema["inputDatasets"]["forecastingBaseHistory"]["idField"]    
    pp_params["location_field"] = data_schema["inputDatasets"]["forecastingBaseHistory"]["locationField"]  
    pp_params["item_field"] = data_schema["inputDatasets"]["forecastingBaseHistory"]["itemField"]     
    pp_params["epoch_field"] = data_schema["inputDatasets"]["forecastingBaseHistory"]["epochField"]   
    pp_params["target_field"] = data_schema["inputDatasets"]["forecastingBaseHistory"]["targetField"]  
    
    # sp_events data fields
    pp_params["se_epoch_field"] = data_schema["inputDatasets"]["forecastingBaseSpecialEvents"]["epochField"] 
    pp_params["event_field"] = data_schema["inputDatasets"]["forecastingBaseSpecialEvents"]["eventField"]  
    pp_params["window_lower"] = data_schema["inputDatasets"]["forecastingBaseSpecialEvents"]["windowLower"]  
    pp_params["window_upper"] = data_schema["inputDatasets"]["forecastingBaseSpecialEvents"]["windowUpper"]  
  
    
    # dataset specs
    pp_params["forecast_granularity"] = data_schema["datasetSpecs"]["forecastGranularity"] 
    pp_params["forecast_horizon_length"] = data_schema["datasetSpecs"]["forecastHorizonLength"] 
    pp_params["epoch_boundary"] = data_schema["datasetSpecs"]["epochBoundary"] 
    pp_params["missing_value_tag"] = data_schema["datasetSpecs"]["missingValueTag"] 
    
    
    # pprint.pprint(pp_params)    
    return pp_params



def get_epoch_start_time(epoch, fcst_granularity,  epochBoundary): 
    
    fcst_granularity = fcst_granularity.lower()
    epochBoundary = str(epochBoundary).lower()
    
    if fcst_granularity != 'weekly':
        epochBoundary = int(epochBoundary) 
    
    if epochBoundary not in BOUNDARIES[fcst_granularity]: 
        raise ValueError(f'''For {fcst_granularity} granularity, 
                         epochBoundary must be one of {BOUNDARIES[fcst_granularity]}. 
                         Given value {epochBoundary}''')
        
    e_weekday = WEEKLY_BOUNDARIES[epoch.weekday()]
    e_hours = epoch.hour     
    e_minutes = epoch.minute            
    e_seconds = epoch.second 
    e_yr = epoch.year
    e_mt = epoch.month
    e_dom = epoch.day
        
    if fcst_granularity == "hourly":    
        if e_minutes < epochBoundary: 
            delta = 60 - (epochBoundary - e_minutes)
        else: 
            delta = e_minutes - epochBoundary                    
        new_epoch = epoch + timedelta(minutes=-delta) + timedelta(seconds=-e_seconds) 
        return new_epoch
        
    elif fcst_granularity == "daily":          
        if e_hours < epochBoundary: 
            delta = 24 - (epochBoundary - e_hours)
        else: 
            delta = e_hours - epochBoundary     
        new_epoch = epoch + timedelta(hours=-delta) + timedelta(minutes=-e_minutes) + timedelta(seconds=-e_seconds) 
        return new_epoch
    
    elif fcst_granularity == "weekly":
        delta = WKDAY_DELTAS[(epochBoundary, e_weekday)]
        new_epoch = epoch + timedelta(days=-delta) + timedelta(hours=-e_hours) \
            + timedelta(minutes=-e_minutes) + timedelta(seconds=-e_seconds)  
        return new_epoch
    
    elif fcst_granularity == "monthly":       
        if e_dom < epochBoundary: 
            epoch_date = date(e_yr, e_mt, epochBoundary) + relativedelta(months=-1)
        else: 
            epoch_date = date(e_yr, e_mt, epochBoundary)  
        new_epoch = datetime.combine(epoch_date, datetime.min.time())
        return new_epoch
    
    elif fcst_granularity == "yearly":     
        boundary_date = date(e_yr, epochBoundary, 1)
        if epoch < boundary_date: 
            epoch_date = boundary_date + relativedelta(year=-1)
        else: 
            epoch_date = boundary_date
        new_epoch = datetime.combine(epoch_date, datetime.min.time()) 
        return new_epoch
    else: 
        raise ValueError(f"Unrecognized forecast granularity: {fcst_granularity}")
    