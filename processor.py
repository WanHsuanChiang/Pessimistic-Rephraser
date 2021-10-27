# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 04:49:25 2021

@author: ezgtt
"""

#import nltk
#nltk.download('punkt')
import preprocessor as p # tweet-preprocessor
from nltk import tokenize
from transformers import XLNetTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import torch

import param

def process(tweets,add_special_token=True):
  p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION) # tweet-preprocessor
  
  new_tweets = []
  for tweet in tweets:
    tweet = p.clean(tweet)
    sentences = tokenize.sent_tokenize(tweet)
    if add_special_token:
        tweet_with_token = ' [SEP]'.join(sentences) + ' [SEP] [CLS]'
        new_tweets.append(tweet_with_token)
    else:
        new_tweets.append(tweet)

  return new_tweets


def get_model_data(tweets, labels , maxlen, batch_size, add_special_token = True):
    
    if add_special_token:
        tweets = process(tweets)
    
    tokenizer = XLNetTokenizer.from_pretrained(param.MODEL, do_lower_case=True)
    
    tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]
    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_tweets]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen= maxlen, dtype="long", truncating="post", padding="post")
    
    # Create attention masks
    attention_masks = []
    
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)
      
    # Convert all of our data into torch tensors, the required datatype for our model
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return data, sampler, dataloader