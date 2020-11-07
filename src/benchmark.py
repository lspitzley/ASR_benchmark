'''
Use settings.ini to configure the benchmark.
'''
import logging
import configparser
import glob
import os
import time
import multiprocessing
import codecs
import transcribe
import metrics
import pandas as pd

from functools import partial

FORMAT = "[%(filename)s:%(lineno)s %(funcName)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
# logging.getLogger(__name__)

#%%
def get_settings(settings_filepath = 'settings.ini'):
    """ load in the settings file """

    settings = configparser.ConfigParser()

    settings.read(settings_filepath)

    return settings


#%%

def main(settings):

    # Load setting file


    asr_systems = settings.get('general', 'asr_systems').split(',')
    data_folders = settings.get('general', 'data_folders').split(',')
    results_folder = settings.get('general', 'results_folder')
    supported_speech_file_types = sorted(['flac', 'mp3', 'ogg', 'wav', 'json'])

    logging.info('asr_systems: %s', asr_systems)
    logging.info('data_folders: %s', data_folders)

    for data_folder in data_folders:
        logging.info('Working on data folder "%s"', data_folder)
        speech_file_type = settings.get('general', 'speech_file_type')

        # Automatically detect the speech file type.
        # Heuristic: the detected speech file type is the one that has the more speech files in data_folder
        #            e.g., if in data_folder there are 10 mp3s and 25 flacs, then choose flac
        if speech_file_type == 'auto':
            maximum_number_of_speech_files = 0
            detected_speech_file_type = None
            for supported_speech_file_type in supported_speech_file_types:
                potential_speech_filepaths = sorted(glob.glob(os.path.join(data_folder, '*.{0}'.format(supported_speech_file_type))))
                if maximum_number_of_speech_files < len(potential_speech_filepaths):
                    maximum_number_of_speech_files = len(potential_speech_filepaths)
                    detected_speech_file_type = supported_speech_file_type
            speech_file_type = detected_speech_file_type
            logging.debug('Detected speech file type: %s', speech_file_type)
            if detected_speech_file_type is None:
                raise ValueError('You have set speech_file_type to be "auto" in {0}. We couldn\'t detect any speech file. Speech file extensions should be {1}.'
                                 .format('settings_filepath', supported_speech_file_types))

        if speech_file_type not in supported_speech_file_types:
            raise ValueError('You have set speech_file_type to be "%s" in %s.'
                             'This is invalid. speech_file_type should be'
                             'flac, ogg, mp3, or wav.',
                             speech_file_type, 'settings_filepath')

        speech_filepaths = sorted(glob.glob(os.path.join(data_folder, '*.{0}'.format(speech_file_type))))

        if settings.getboolean('general', 'transcribe'):

            # Make sure there are files to transcribe
            if not speech_filepaths:
                raise ValueError('There is no file with the extension "%s"  in the folder "%s"',
                                 speech_file_type, data_folder)

            # Transcribe
            logging.debug('\n### Call the ASR engines')
            for speech_file_number, speech_filepath in enumerate(speech_filepaths):
                # Convert the speech file from FLAC/MP3/Ogg to WAV
                if speech_file_type in ['flac', 'mp3', 'ogg']:
                    from pydub import AudioSegment
                    logging.debug('speech_filepath: %s', speech_filepath)
                    sound = AudioSegment.from_file(speech_filepath, format=speech_file_type)
                    new_speech_filepath = speech_filepath[:-len(speech_file_type)-1]+'.wav'
                    sound.export(new_speech_filepath, format="wav")
                    speech_filepath = new_speech_filepath

                # Transcribe the speech file
                all_transcription_skipped = True
                for asr_system in asr_systems:
                    # logging.debug('sending call to transcription')
                    transcription, transcription_skipped = transcribe.transcribe(speech_filepath, asr_system, settings, results_folder, save_transcription=True)
                    all_transcription_skipped = all_transcription_skipped and transcription_skipped

                # If the speech file was converted from FLAC/MP3/Ogg to WAV, remove the WAV file
                if speech_file_type in ['flac', 'mp3', 'ogg']:
                    os.remove(new_speech_filepath)

                if not all_transcription_skipped:
                    time.sleep(settings.getint('general','delay_in_seconds_between_transcriptions'))

        if settings.getboolean('general','evaluate_transcriptions'):
            # Evaluate transcriptions
            all_texts = {}
            logging.info(('\n#######\n'
                          'Final evaluation of all the ASR engines based on their predicted jurisdictions\n'
                          '########\n'))

            for asr_system in asr_systems:
                all_texts[asr_system] = {}




                # get new results folder:
                # results_filepaths = sorted(glob.glob(os.path.join(results_folder, '*.{0}'.format(speech_file_type))))

                # store results for individual files
                result_stats_file = os.path.join(data_folder, 'stats', asr_system + '_' 'results.csv')
                logging.info('saving results to %s', result_stats_file)
                with open(result_stats_file, 'w') as f:
                    f.writelines('id,changes,corrects,substitutions,insertions,deletions,n_tokens_gold\n')

                start_time = time.time()
                pool = multiprocessing.Pool()

                store_words = settings.getboolean('general', 'save_word_level')
                logging.info("Starting processes.")

                results = pool.map(partial(get_transcript_wer,
                                           word_level_save=store_words,
                                           results_folder=results_folder,
                                           asr_system=asr_system,
                                           settings=settings), speech_filepaths)
                print(results)

                pool.terminate()
                pool.join()
                with open(result_stats_file, 'a') as f:
                    for row in results:
                        try:
                            f.write("%s\n" % row)
                        except TypeError:
                            logging.warning('row %s not in results', row)
                logging.info('processed wer in %s seconds', time.time() - start_time)


def get_transcript_wer(speech_filepath, results_folder, asr_system, settings, word_level_save=False):
    """
    attempt to functionalize the accuracy computation
    to allow parallel computing.

    currently uses a brute force approach to arguments
    change this later.
    """
    #results_folder = args['results_folder']
    #asr_system = args['asr_system']


    edit_types = ['corrects', 'deletions', 'insertions', 'substitutions', 'changes']
    number_of_edits = {}

    for edit_type in edit_types:
        number_of_edits[edit_type] = 0

    filename = os.path.basename(speech_filepath)
    results_filepath = os.path.join(results_folder, filename)

    logging.debug('results filename %s', results_filepath)
    predicted_transcription_filepath_base = '.'.join(results_filepath.split('.')[:-1]) + '_'  + asr_system
    predicted_transcription_txt_filepath = predicted_transcription_filepath_base  + '.txt'

    if not os.path.isfile(predicted_transcription_txt_filepath):
        logging.warning('predicted transcription not found %s', speech_filepath)
        return

    predicted_transcription = load_predicted_transcription(predicted_transcription_txt_filepath, settings)
    if len(predicted_transcription) == 0:
        logging.warning('transcript of length 0: %s', speech_filepath)
        return

    try:
        # gold_transcription_filepath_base = '.'.join(speech_filepath.split('.')[:-1]) + '_'  + 'gold'
        gold_transcription_filepath_base = '.'.join(speech_filepath.split('.')[:-1]) + '_'  + 'all'
        gold_transcription_filepath_text = gold_transcription_filepath_base  + '.txt'
        gold_transcription = codecs.open(gold_transcription_filepath_text, 'r', settings.get('general','gold_transcription_encoding')).read()
    except FileNotFoundError:
        logging.error('file not found: %s', results_filepath)
        return


    gold_transcription = metrics.normalize_text(gold_transcription, lower_case=True, remove_punctuation=True,write_numbers_in_letters=True)
    predicted_transcription = metrics.normalize_text(predicted_transcription, lower_case=True, remove_punctuation=True,write_numbers_in_letters=True)
    logging.debug('found file %s', speech_filepath)

    #print('\npredicted_transcription\t: {0}'.format(predicted_transcription))
    #print('gold_transcription\t: {0}'.format(gold_transcription))

    gold_transcript_tokens = gold_transcription.split(' ')
    wer, lines = metrics.wer(gold_transcript_tokens, predicted_transcription.split(' '), word_level_save)
    #print('wer: {0}'.format(wer))

    #if len(predicted_transcription) == 0: continue

    number_of_tokens_in_gold = len(gold_transcript_tokens)
    for edit_type in edit_types:
        number_of_edits[edit_type] += wer[edit_type]

    # store word-level transcripts
    if word_level_save:
        save_word_level(results_folder, filename, asr_system, lines)
    new_row = filename + ',' + wer_stats(wer, number_of_tokens_in_gold)
    logging.debug('new row for %s is %s', speech_filepath, new_row)

    return new_row

def save_word_level(results_folder, filename, asr_system, lines):
    """
    write csv with word-level analysis
    """
    words_filepath = os.path.join(results_folder, 'words', filename)
    words_filepath_base = '.'.join(words_filepath.split('.')[:-1]) + '_'  + asr_system
    words_csv_filepath = words_filepath_base  + '.csv'

    logging.debug('lines in %s: %d', filename, len(lines))

    with open(words_csv_filepath, 'w+') as f:
        # need to reverse, since algorithm works from end to start
        for line in reversed(lines):
            f.write("%s\n" % line)

def load_predicted_transcription(predicted_transcription_txt_filepath, settings):
    predicted_transcription = codecs.open(predicted_transcription_txt_filepath, 'r', settings.get('general','predicted_transcription_encoding')).read().strip()
    return predicted_transcription

def wer_stats(wer, n_tokens):
    """
    format WER stats for a csv file output.
    """
    logging.info('wer for doc %s, n_tokens %s', wer, n_tokens)
    new_row = ",".join(map(str, wer.values())) + ',' + str(n_tokens)
    #logging.info('result row %s', new_row)
    return new_row

#%%
def rename_wer_keys(stats_dict, asr_name):
    """
    Rename the keys for each system so that they
    can be added to the dataframe. 

    Parameters
    ----------
    stats_dict : dict
        original dictionary with standard wer names.
    
    asr_name : str
        string to append to new names

    Returns
    -------
    dictionary with renamed keys.

    """
    new_dict = {}
    for key in stats_dict:
        print(key)
        new_key = '_'.join([key, asr_name])
        new_dict[new_key] = stats_dict[key]

    return new_dict

#%% 

def csv_eval_main(settings):
    """ 
    run program on csv columns and evaluate accuracy. 
    
    This code is pretty ugly but works for this one dataset. 
    
    It could be broken down into several functions and made more general.
    
    """
    csv_file =  settings.get('general', 'csv_file')
    logging.info('Reading CSV file %s', csv_file)
    
    transcripts_df = pd.read_csv(csv_file)
    
    # TODO accept column names as settings or args 
    
    # normalize text (replace old columns)
    transcripts_df['gold_std'] = transcripts_df['Ground_truth'].apply(func=metrics.normalize_text, args=(True, True, True,))
    transcripts_df['ibm_std'] = transcripts_df['Watson_ASR'].apply(func=metrics.normalize_text, args=(True, True, True,))
    transcripts_df['goog_std'] = transcripts_df['Goog_ASR'].apply(func=metrics.normalize_text, args=(True, True, True,))
    transcripts_df['amzn_std'] = transcripts_df['Amzn_ASR'].apply(func=metrics.normalize_text, args=(True, True, True,))
    
    # run metrics.wer
    # metrics.wer(gold_transcript_tokens, predicted_transcription.split(' '), word_level_save)
    out = transcripts_df.apply(lambda x: metrics.wer(x['gold_std'].split(' '), x['ibm_std'].split(' '), True), axis=1)
    print(out.apply(lambda x: print(x[0])))
    ibm_out = out.apply(lambda x: x[0].copy())

    
    # rename keys (changes_ibm, corrects_ibm, etc.)
    ibm_keys = ibm_out.apply(lambda x: rename_wer_keys(x, 'ibm'))
    
    
    # run in loop instead of one-by-one
    asr_list = ['ibm_std', 'goog_std', 'amzn_std']
    for asr in asr_list:
        out = transcripts_df.apply(lambda x: metrics.wer(x['gold_std'].split(' '), x[asr].split(' '), True), axis=1)
        print(out.apply(lambda x: print(x[0])))
        asr_out = out.apply(lambda x: x[0].copy())

        # rename keys (changes_ibm, corrects_ibm, etc.)
        asr_keys = asr_out.apply(lambda x: rename_wer_keys(x, asr))
        
        # tabulate results
        
        new_df = asr_keys.apply(pd.Series)
        transcripts_df = pd.concat([transcripts_df, new_df], axis=1, sort=False)
        
    

    
    # save file
    transcripts_df.to_csv('../data/wer_csv_out.csv')
    
    return transcripts_df


def txt_eval_main(settings_ini):
    """ evaluate data from txt file """
     

    # read data
    data_folder = settings_ini.get('general', 'data_folders')
    txt_gold = ''
    txt_ibm = ''
        
            
    with open(os.path.join(data_folder, '008SB_gold.txt'), 'r', errors="ignore", encoding='utf-8') as f:
        txt_gold = f.read()
        
    with open(os.path.join(data_folder, '008SB_ibm_post.txt'), 'r', errors="ignore", encoding='utf-8') as f:
        txt_ibm = f.read()

        
    # preprocess
    # print(txt_gold)
    gold_clean = metrics.normalize_text(txt_gold, True, True, True)
    asr_clean = metrics.normalize_text(txt_ibm, True, True, True)
    print('gold_len', len(gold_clean.split(' ')), 'asr_len', len(asr_clean.split(' ')))    
    
    
    # compute accuracy
    
    acc = metrics.wer(gold_clean.split(' '), asr_clean.split(' '), debug=True)
    
    # show results

    print('results', acc[0])
    
    

#%%

if __name__ == "__main__":
    start = time.time()
    logging.info('program started')
    settings_ini = get_settings()
    print(settings_ini.getboolean('general', 'from_csv')==False)
    if settings_ini.getboolean('general', 'from_csv'):
        transcripts_df = csv_eval_main(settings_ini)
    elif settings_ini.getboolean('general', 'from_txt'):
        transcripts_df = txt_eval_main(settings_ini)
    else:
        main(settings_ini)
    #cProfile.run('main()') # if you want to do some profiling
    logging.info('done. took %.6f seconds', time.time() - start)
    print('done. time in seconds: ', time.time() - start)
