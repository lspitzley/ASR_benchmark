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

import re
import logging
import pandas as pd
import os

FORMAT = "[%(filename)s:%(lineno)s %(funcName)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



#%%%

def utterance_iterator(transcript, word_list, text_col='text'):
    """ 
    goes through each line of the transcript
    and matches the text in the word list. 
    """
    position = 0
    for row in transcript.itertuples():
        print(getattr(row, 'speakerName'))
        text = getattr(row, text_col)
        # logging.debug('text: %s', text)
        start_position = position
        position = word_matcher(text, word_list, position)
        
        # do something to store the word-level with metadata
        # based on the start position and end position
        
        # can use a pandas summarization to get stats by type
        
        # might need to convert the number words again

#%%%
# debug stuff
text = 'Greetings, and welcome to Genius Brands International Third Quarter 2018 Earnings Conference Call. At this time, all participants are in a listen-only mode. A question-and-answer session will follow the formal presentation. [Operator Instructions] As a reminder, this conference is being recorded. I would now like to turn the conference to Michael Porter, with Porter, LeVay & Rose. Please, you may begin.'
def word_matcher(text, word_list, position):
    text = re.sub(r'\[.+\]', ' ', text)
    text = re.sub(r'[^\w\d\- ]+', ' ', text) # clean the text
    text_list = re.split(r'[\s-]+', text.strip().lower())
    # logging.debug(word_list)
    #%%
    i = 0
    match_count = 0
    miss_count = 0
    cur_pos = last_match = position
    while (i < len(text_list)) and (cur_pos - position < (2 * len(text_list))) and (cur_pos < len(word_list.index)):
        base_word = text_list[i] 
        ref_word = re.sub(r'[^\w\d\*]', '', word_list.loc[cur_pos, 'REF'])
        
        # if it gets too far ahead, move one forward and reset cur_pos
        if cur_pos - last_match > 6:
            cur_pos = last_match + 1
            i += 1
            logging.debug('resetting count, too many misses')
            continue
        
        #logging.debug('at %d', cur_pos)
        if ref_word == '****':
            logging.debug('found insertion at cur_pos %s', cur_pos)
            cur_pos += 1
            last_match += 1
            continue
        if ref_word == base_word:
            logging.debug('match at i = %s, cur_pos = %s is %s', i, cur_pos, base_word)
            i += 1
            last_match = cur_pos
            cur_pos += 1
            match_count += 1
        elif text_list[i].isnumeric():
            logging.debug('numeric at i = %d, cur_pos = %d', i, cur_pos)
            i += 1
            continue
        else:
            logging.debug('no match at i = %d, cur_pos = %d. Actual = %s REF = %s', i, cur_pos, base_word, ref_word)
            cur_pos += 1
            miss_count += 1
            continue
        



#%%
    return cur_pos

def test_word_matcher():
    text = subset.loc[169970, 'text']
    print(text)    




#%%%%
ground_truth_file = '../data/ground_truth/call_df_likely_audio.csv'

ground_truth = pd.read_csv(ground_truth_file)



#%%
word_files_dir = '../data/eval_test/results/words/'
word_files = os.listdir(word_files_dir)

#%%
for file in word_files:

#%%
    #file = word_files[1]
    sa_id = int(file.split('_')[0])
    # trim the first row, since it contains the speaker information
    subset = ground_truth.loc[ground_truth['SA_ID'] == sa_id,][1:]
    # print((word_files_dir, file))
    word_list = pd.read_csv(os.path.join(word_files_dir, file), delimiter='\t', encoding='latin-1')
    #print(subset.loc[169970, 'text'])    
    utterance_iterator(subset, word_list)
    