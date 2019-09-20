#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:03:55 2019

This script takes a CSV file with the ground truth text and accompanying
meta data and aligns it with the results of word error computation. This will
allow statistics more granular than an entire document, which is
especially useful in situations where there multiple speakers or distinct
events in the audio files.

@author: leespitzley
"""

import logging
import pandas as pd
import os

FORMAT = "[%(filename)s:%(lineno)s %(funcName)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

#%%%
def word_matcher():
    pass




def utterance_iterator(transcript, word_list, text_col='text'):
    """ 
    goes through each line of the transcript
    and matches the text in the word list. 
    """
    
    for row in transcript.itertuples():
        print(getattr(row, 'speakerName'))
        text = getattr(row, text_col)
        logging.debug('text: %s', text)




#%%%%
ground_truth_file = '../data/ground_truth/call_df_likely_audio.csv'

ground_truth = pd.read_csv(ground_truth_file)



#%%
word_files_dir = '../data/eval_test/results/words/'
word_files = os.listdir(word_files_dir)

#%%
for file in word_files:
    pass

#%%
    file = word_files[0]
    sa_id = int(file.split('_')[0])
    # trim the first row, since it contains the speaker information
    subset = ground_truth.loc[ground_truth['SA_ID'] == sa_id,][1:]
    word_list = pd.read_csv(os.path.join(word_files_dir, file), delimiter='\t')
    
    utterance_iterator(subset, word_list)
    