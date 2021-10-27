import pandas as pd
import numpy as np
from fast_ml.model_development import train_valid_test_split

def get_opt(pess_threshold = -1, opt_threshold = 1):    
    path = 'data/tweets_annotation.csv'
    df = pd.read_csv(path)
  
    if pess_threshold == opt_threshold:
      conditions = [
        (df['AverageAnnotation'] <= pess_threshold),
        (df['AverageAnnotation'] > opt_threshold)
      ]
      values = ['pessimistic','optimistic']
      target =[0,1]
    elif pess_threshold < opt_threshold:
      conditions = [
        (df['AverageAnnotation'] >= opt_threshold),
        (df['AverageAnnotation'] <= pess_threshold),
        (df['AverageAnnotation'] < opt_threshold) & (df['AverageAnnotation'] > pess_threshold)
      ]
      values = ['optimistic','pessimistic','neutral']
      target = [1,0,None]
    else:
      print('The pessimistic error is greater than optimistic threshold')
      return
    
    df['Label'] = np.select(conditions,values)
    df['Target'] = np.select(conditions,target)  
    filtered_df = df.loc[(df['Label'] != 'neutral'),df.columns]
    filtered_df.astype({'Target': 'int32'}).dtypes
    filtered_df = filtered_df.reset_index(drop=True)
  
    return filtered_df

def split(df,X,target,train_size, valid_size, test_size):
    
    removed_columns = [column for column in df.columns if column not in [X,target] ]
    new_df = df.drop(columns = removed_columns)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(new_df, target = target, train_size=train_size, valid_size=valid_size, test_size=test_size)
    return list(X_train[X]), list(y_train), list(X_valid[X]), list(y_valid), list(X_test[X]), list(y_test)