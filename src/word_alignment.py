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
import numpy as np
import pandas as pd
import os

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
FORMAT = "[%(filename)s:%(lineno)s %(funcName)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT, filename='../logs/alignment.log')



#%%%

def utterance_iterator(transcript, word_list, text_col='text'):
    """ 
    goes through each line of the transcript
    and matches the text in the word list. 
    """

    position = 0
    sentences = []
    for row in transcript.itertuples():
        #print(getattr(row, 'speakerName'))
        text = getattr(row, text_col)
        # logging.debug('text: %s', text)
        start_position = position
        position, sentence_ends = word_matcher(text, word_list, position)
        logging.info('started at %s, ended at %s. Found sentences at %s.', 
                     start_position, position, sentence_ends)
        
        # do something to store the word-level with metadata
        # based on the start position and end position
        
        sentences.append(merge_word_list(word_list, start_position, position, sentence_ends))
        
        logging.info('resulted in predicted text %s', sentences[len(sentences)-1])
        # can use a pandas summarization to get stats by type
        
        # might need to convert the number words again


    return sentences

#%%%
# debug stuff
text = 'Thank you sir. [Operator Instructions] Your first question will be from John Inch of Gordon Haskett. Please go ahead.'
position = 3919
def word_matcher(text, word_list, position):
    text += ' ' # add an extra space to the end of text to find the last sentence
    text = re.sub(r'[\'\"\’]', '', text)
    text = re.sub(r'[\.]\s+', ' . ', text) # replace periods to reconstruct sentences
    text = re.sub(r'[\?]\s+', ' ? ', text) # replace periods to reconstruct sentences
    text = re.sub(r'[\!]\s+', ' ! ', text) # replace periods to reconstruct sentences
    text = re.sub(r'\[|\]', ' ', text) # remove operator subtext
    text = re.sub(r'[^\w\d\-\.\?\! ]+', ' ', text) # clean the text
    text_list = re.split(r'[\s-]+', text.strip().lower())
    # logging.debug(word_list)
    
    # assemble an utterance of hypothesis words
    sentence_ends = {}
    
    i = 0
    match_count = 0
    miss_count = 0
    cur_pos = last_match = position
    while (i < len(text_list)) and (cur_pos - position < (2 * len(text_list))) and (cur_pos < len(word_list.index)):
        base_word = text_list[i] 
        ref_word = re.sub(r'[^\w\d\*]', '', word_list.loc[cur_pos, 'REF'])
        # hyp_word = re.sub(r'[^\w\d\*]', '', word_list.loc[cur_pos, 'HYP'])
        
        # if at the end of a sentence, annotate the hypothesis text.
        if end_of_sentence(base_word):
            sentence_ends[cur_pos] = base_word
            i += 1
            continue
        
        # if it gets too far ahead, move one forward and reset cur_pos
        if cur_pos - last_match > 8:
            cur_pos = last_match + 1
            i += 1
            logging.debug('resetting count, too many misses')
            continue
        
        #logging.debug('at %d', cur_pos)
        if ref_word == '****':
            logging.debug('found insertion at cur_pos %s', cur_pos)
            cur_pos += 1
            last_match += 1
            position += 1 # don't count insertions against the ovral transcript length
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
    
    logging.debug('last word is %s', text_list[len(text_list)-1])
    if end_of_sentence(text_list[len(text_list)-1].strip()):
        sentence_ends[cur_pos] = text_list[len(text_list)-1].strip()
    logging.info('finished utterance %s at position i %d', text, i)
    return cur_pos, sentence_ends

def test_word_matcher():
    text = subset.loc[169970, 'text']
    print(text)    


def end_of_sentence(word):
    """ reusable check for end of sentence """
    if word in ['.', '?', '!']:
        return word
    return ''

def merge_word_list(word_list, start_pos, end_pos, sentences):
    full_prediction = ''
    for i in range(start_pos, end_pos):
        full_prediction += " " + str(word_list.iloc[i]['HYP'])
        # look if it is the end of a sentence
        if i+1 in sentences:
            full_prediction += str(sentences[i+1])
    
    # add last period
    return full_prediction.strip()

#%%%%
ground_truth_file = '../data/ground_truth/call_df_likely_audio.csv'

ground_truth = pd.read_csv(ground_truth_file)



#%%
word_files_dir = '../data/eval_test/results/words/'
word_files = os.listdir(word_files_dir)

#%%
for file in word_files:

#%%
    file = word_files[1]
    sa_id = int(file.split('_')[0])
    # trim the first row, since it contains the speaker information
    subset = ground_truth.loc[ground_truth['SA_ID'] == sa_id,]
    logging.info('first speaker in %s is %s', file, subset.iloc[0]['speakerName'])
    if pd.isnull(subset.iloc[0]['speakerName']):
        subset = subset[1:]
    # print((word_files_dir, file))
    word_list = pd.read_csv(os.path.join(word_files_dir, file), delimiter='\t', encoding='latin-1')
    #print(subset.loc[169970, 'text'])    
    logging.info('working on %s', file)
    
    subset['sentences_watson'] = utterance_iterator(subset, word_list)
    
    subset.to_csv('../data/eval_test/results/transcripts/' + str(sa_id) + '_utterance-level.csv')
    