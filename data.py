# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 20:56:13 2021

@author: ezgtt
"""

import pandas as pd
import numpy as np

def get_tweet(label= 'all'):
    
    '''
    Return Tweet data
    Columns:
        
    Tweet
    Username
    AverageAnnotation
    Label: ['optimistic','pessimistic','neutral']
    '''
    
    ## open file
    path = 'data/optimism-twitter-data/tweets_annotation.csv'
    df = pd.read_csv(path)
    
    ## add label
    # if score > 1: optimistic
    # if score < -1: pessimistic
    # if -1 <= score <= 1: neutral
    conditions = [
    (df['AverageAnnotation'] > 1),
    (df['AverageAnnotation'] < -1),
    (df['AverageAnnotation'] <= 1) & (df['AverageAnnotation'] >= -1)
    ]
    
    values = ['optimistic','pessimistic','neutral']
    df['Label'] = np.select(conditions,values)
    
    ## return dataframe
    if label == 'all':
        return df
    else:
        filtered_df = df.loc[(df['Label']== label),df.columns]
        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df