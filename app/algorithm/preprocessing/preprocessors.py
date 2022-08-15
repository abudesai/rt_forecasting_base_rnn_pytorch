
import numpy as np, pandas as pd
import sys , os
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import algorithm.preprocessing.preprocess_utils as pp_utils



class DummyColumnCreator(BaseEstimator, TransformerMixin):
    ''' creates a dummy feature if the feature doesn't exist. This is used to add the target
    feature for the prediction task. '''

    def __init__(self, col_name, assigned_val ):
         self.col_name = col_name
         self.assigned_val = assigned_val

    def fit(self, X, y=None): return self

    def transform(self, X):
        if self.col_name not in X.columns: 
            X[self.col_name] = self.assigned_val
        return X
        
        


class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, fields, cast_type):
        super().__init__()
        if not isinstance(fields, list):
            self.fields = [fields]
        else:
            self.fields = fields
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.fields if col in data.columns] 
        for col in applied_cols: 
            data[col] = data[col].apply(self.cast_type)
        return data


class StringTypeCaster(TypeCaster):  
    ''' Casts categorical features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, fields): 
        super(StringTypeCaster, self).__init__(fields, str)
        

class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, fields):
        super(FloatTypeCaster, self).__init__(fields, float)
        

class DateTimeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, fields):
        super().__init__()
        if not isinstance(fields, list):
            self.fields = [fields]
        else:
            self.fields = fields
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):      
        for field in self.fields:         
            data[field] = pd.to_datetime(data[field])
        return data



class MissingValueTagger(BaseEstimator, TransformerMixin):  
    def __init__(self, id_fields, value_field, missing_field, missing_tag):
        super().__init__()
        self.id_fields = id_fields
        self.value_field = value_field
        self.missing_field = missing_field
        self.missing_tag = missing_tag
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        if self.missing_tag == "": return data
        
        # find epochs with missing values
        idx = data[self.value_field] == self.missing_tag        
        
        # if idx.sum() == 0 : return data
                        
        # initiate the missing field with zeros
        data[self.missing_field] = 1.
        
        # set missing flag to 1
        data.loc[idx, self.missing_field] = 0.   
        
        # fill the missing values with a very low value. 
        # we set it to be observed_min - observed_range. 
        non_missing =  data.loc[~idx]
        
        non_missing[self.value_field] = non_missing[self.value_field].astype(np.float32)
        
        # ranges_df = non_missing.groupby(self.id_fields, as_index=False)[self.value_field].apply(lambda x: x.max() - x.min())
        ranges_df = non_missing.groupby(self.id_fields, as_index=False)[self.value_field].min()
        ranges_df.rename(columns={self.value_field: "_range"}, inplace=True)
        
        data = data.merge(ranges_df, on=self.id_fields, how='left')
        
        data.loc[idx, self.value_field] = -data.loc[idx, "_range"]
        data.loc[idx, self.value_field] = 0
        
        data[self.value_field] = data[self.value_field].fillna(0.)
        
        del data["_range"]
        return data


class EpochResetter(BaseEstimator, TransformerMixin):
    ''' Updates the epoch time field represent the epoch start time as per forecast granularity. 
        For example, support forecast granularity is daily, and epoch_boundary is 9 
        (i.e. day starts at 9 am and ends at 8:59:59 am)
        Then all epochs between 9 am on August 1st to 8:59:59 am August 2nd will show the
        same epoch of August 1st 9 am.    
    '''
    def __init__(self, time_column, time_granularity,  epochBoundary):
        super().__init__()
        self.time_column = time_column
        self.time_granularity = time_granularity
        self.epochBoundary = epochBoundary


    def fit(self, data, y=None): return self

    def transform(self, data):  
        data[self.time_column] = data[self.time_column].apply(
            lambda e: pp_utils.get_epoch_start_time(e, self.time_granularity,  self.epochBoundary) )  
        
        return data
        
        

class ValueAggregator(BaseEstimator, TransformerMixin):
    ''' Aggregates time-series values to the given time interval level.'''
    def __init__(self, group_by_cols, aggregated_columns):
        super().__init__()
        if not isinstance(group_by_cols, list):
            self.group_by_cols = [group_by_cols]
        else:
            self.group_by_cols = group_by_cols
        if not isinstance(aggregated_columns, list):
            self.aggregated_columns = [aggregated_columns]
        else:
            self.aggregated_columns = aggregated_columns    

    def fit(self, data, y=None): return self

    def transform(self, data): 
        groupby_cols = [ col for col in self.group_by_cols if col in data.columns]
        agg_cols = [ col for col in self.aggregated_columns if col in data.columns]
        data = data.groupby(by=groupby_cols, as_index=False)[agg_cols].sum()  
        return data


class MissingIntervalFiller(BaseEstimator, TransformerMixin):
    ''' Adds missing time intervals in a time-series dataframe.     '''

    def __init__(self, id_columns, time_column, value_columns, time_granularity):
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

        self.time_granularity = str(time_granularity).lower()

        self.max_epoch = None
        self.relative_delta_args = None
    
    def fit(self, X, y=None): return self # do nothing in fit
        

    def transform(self, X):      
        
        min_time = X[self.time_column].min()
        max_time = X[self.time_column].max()      
        # print(min_time, max_time)  
        delta_dict = {}
        if self.time_granularity == 'yearly':
            delta_dict = {"years": 1} 
        elif self.time_granularity == 'monthly':
            delta_dict = {"months": 1}
        elif self.time_granularity == 'weekly':
            delta_dict = {"weeks": 1}
        elif self.time_granularity == 'daily':
            delta_dict = {"days": 1}
        elif self.time_granularity == 'hourly':
            delta_dict = {"hours": 1}
        else: 
            raise Exception(f'''Unrecognized time granularity: {self.time_granularity}. 
                            Must be one of ['hourly', 'daily', 'weekly', 'monthly', 'yearly'].''')
        
        
        curr_time = min_time
        all_time_ints = []
        while curr_time <= max_time: 
            all_time_ints.append(curr_time)
            curr_time += relativedelta(**delta_dict)
        
        # save this for inverse_transformation on predictions
        self.relative_delta_args = delta_dict
        self.max_epoch = max(all_time_ints)
        
        
        # create df of all time intervals
        full_intervals_df = pd.DataFrame(data = all_time_ints, columns = [self.time_column])   

        
        if len(self.id_columns) > 0: 
            # get unique id-var values from original input data
            id_cols_df = X[self.id_columns].drop_duplicates()
            dummy_col = ''
        else: 
            dummy_col = '__DUMMYCOL__'
            id_cols_df = pd.DataFrame([[0]], columns=[dummy_col])
        
        # get cross join of all time intervals and ids columns
        full_df = id_cols_df.assign(foo=1).merge(full_intervals_df.assign(foo=1)).drop('foo', 1)

        # merge original data on to this full table
        value_cols = [ col for col in self.value_columns if col in X]
        full_df = full_df.merge(X[self.id_columns + [self.time_column] + value_cols], 
                    on=self.id_columns + [self.time_column], how='left')        
                
        if dummy_col in full_df.columns: del full_df[dummy_col]
        return full_df


class NAFiller(BaseEstimator, TransformerMixin):  
    def __init__(self, fields, fill_val):
        super().__init__()        
        if not isinstance(fields, list):
            self.fields = [fields]
        else:
            self.fields = fields
        self.fill_val = fill_val
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data[self.fields] = data[self.fields].fillna(self.fill_val)
        return data




class SpEventsNeighborhoodMarker(BaseEstimator, TransformerMixin):  
    def __init__(self, event_field, regular_epoch_lbl, epochs_before_field, epochs_after_field,
                 epochs_before_suffix, epochs_after_suffix):
        super().__init__()
        self.event_field = event_field
        self.regular_epoch_lbl = regular_epoch_lbl
        self.epochs_before_field = epochs_before_field
        self.epochs_after_field = epochs_after_field
        self.epochs_before_suffix = epochs_before_suffix
        self.epochs_after_suffix = epochs_after_suffix
    
    def fit(self, X, y=None): return self # do nothing in fit

    def transform(self, data): 
        # data.to_csv("data.csv", index=False)
        # sys.exit()
        num_rows = data.shape[0]
        # after events
        row_num = 0; run_loop = True
        while run_loop: 
            event = data.at[row_num, self.event_field]
            if event != self.regular_epoch_lbl: 
                num_epochs_after = int(data.at[row_num, self.epochs_after_field])
                row_num += 1
                for i_after in range(num_epochs_after):                    
                    new_event = event + self.epochs_after_suffix + str(i_after+1)
                    data.at[row_num, self.event_field] = new_event
                    row_num += 1
                    if row_num >= num_rows: break
            else: 
                row_num += 1            
            if row_num >= num_rows - 1: run_loop = False  # " - 1" because even if last epoch is special, there is no 'after' epoch to it. 
        
        
        # before events
        row_num = num_rows-1; run_loop = True
        while run_loop: 
            event = data.at[row_num, self.event_field]
            if event != self.regular_epoch_lbl and self.epochs_after_suffix not in event: 
                num_epochs_before = int(data.at[row_num, self.epochs_before_field])
                row_num -= 1
                for i_before in range(num_epochs_before):    
                    if self.epochs_after_suffix not in event:              
                        new_event = event + self.epochs_before_suffix + str(i_before+1)
                        data.at[row_num, self.event_field] = new_event
                    row_num -= 1
                    if row_num < 0: break
            else: 
                row_num -= 1            
            if row_num < 1: run_loop = False        # " < 1" because even if first epoch is special, there is no 'before' epoch to it.   
        
        return data



class NAFillerUsingSubstring(BaseEstimator, TransformerMixin):  
    def __init__(self, field_substring, fill_na_val):
        super().__init__()        
        self.field_substring = field_substring
        self.fill_na_val = fill_na_val
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        cols = [ col for col in data.columns if self.field_substring in col]
        data[cols] = data[cols].fillna(self.fill_na_val)
        return data


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type.lower()
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):   
        
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')   
        return X
    
    


class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):  
    def __init__(self, ohe_columns, prefix, fill_na_val, max_num_categories=10): 
        super().__init__()    
        if not isinstance(ohe_columns, list):
            self.ohe_columns = [ohe_columns]
        else:
            self.ohe_columns = ohe_columns
        self.prefix = prefix
        self.max_num_categories = max_num_categories
        self.top_cat_by_ohe_col = {}
        self.fill_na_val = fill_na_val
        
        
    def fit(self, X, y=None):    
        for col in self.ohe_columns:
            if col in X.columns: 
                self.top_cat_by_ohe_col[col] = [ 
                    cat for cat in X[col].value_counts()\
                        .sort_values(ascending = False).head(self.max_num_categories).index
                    ]         
        return self
    
    
    def transform(self, data): 
        df_list = [data]
        cols_list = list(data.columns)
        for col in self.ohe_columns:
            if len(self.top_cat_by_ohe_col[col]) > 0:
                if col in data.columns:                
                    for cat in self.top_cat_by_ohe_col[col]:
                        col_name = self.prefix + col + '_' + str(cat)
                        vals = np.where(data[col] == cat, 1, 0)
                        df = pd.DataFrame(vals, columns=[col_name])
                        df_list.append(df)
                        cols_list.append(col_name)
                else: 
                    raise Exception(f'''
                        Error: Fitted one-hot-encoded column {col}
                        does not exist in dataframe given for transformation.
                        This will result in a shape mismatch for train/prediction job. 
                        ''')
        
        transformed_data = pd.concat(df_list, axis=1, ignore_index=True) 
        transformed_data.columns =  cols_list
        transformed_data[cols_list] = transformed_data[cols_list].fillna(self.fill_na_val)
        return transformed_data


    
   