import pandas as pd
import numpy as np
import re

def get_opt(label= 'all'):
    
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
    
def get_sst():
    
    # https://medium.com/analytics-vidhya/sentiment-analysis-for-text-with-deep-learning-2f0a0c6472b5
    
    path = 'data/stanfordSentimentTreebank/'
    
    # read dictionary into df
    df_data_sentence = pd.read_table(path + 'dictionary.txt')
    df_data_sentence_processed = df_data_sentence['!|0'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})
    # read sentiment labels into df
    df_data_sentiment = pd.read_table(path + 'sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})
    #combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')
    return df_processed_all 



def get_tsa():
    
    path = 'data/Sentiment-Analysis-Dataset/Sentiment Analysis Dataset.csv'
    with open(path,'rb') as file:
        lines = file.readlines()
        #header = lines[0].decode('utf-8').rstrip().split(',')
        header = lines[0].decode('utf-8').split(',')
        data_list = []
        for line in lines[1:]:
            props = line.decode('utf-8').split(',')
            if len(props) > 4:
                sentiment_text = ','.join(props[3:])
                data = props[:3]
                data.append(sentiment_text)
            else:
                data = props
            data_list.append(data)
            # test index = 4286
    df = pd.DataFrame(data_list, columns = header)
    return df