'''
Use settings.ini to configure the benchmark.
'''
import logging
FORMAT = "[%(filename)s:%(lineno)s %(funcName)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
# logging.getLogger(__name__)

import configparser
import glob
import os
import transcribe
import metrics
import time
import threading
import shutil
import codecs







def main():

    # Load setting file
    settings = configparser.ConfigParser()
    settings_filepath = 'settings.ini'
    settings.read(settings_filepath)

    asr_systems = settings.get('general','asr_systems').split(',')
    data_folders = settings.get('general','data_folders').split(',')
    results_folder = settings.get('general','results_folder')
    supported_speech_file_types = sorted(['flac', 'mp3', 'ogg', 'wav', 'json'])

    print('asr_systems: {0}'.format(asr_systems))
    print('data_folders: {0}'.format(data_folders))

    for data_folder in data_folders:
        print('\nWorking on data folder "{0}"'.format(data_folder))
        speech_file_type = settings.get('general','speech_file_type')

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
            print('Detected speech file type: {0}'.format(speech_file_type))
            if detected_speech_file_type is None:
                raise ValueError('You have set speech_file_type to be "auto" in {0}. We couldn\'t detect any speech file. Speech file extensions should be {1}.'
                                 .format(settings_filepath, supported_speech_file_types))

        if speech_file_type not in supported_speech_file_types:
            raise ValueError('You have set speech_file_type to be "{0}" in {1}. This is invalid. speech_file_type should be flac, ogg, mp3, or wav.'.
                             format(speech_file_type, settings_filepath))

        speech_filepaths = sorted(glob.glob(os.path.join(data_folder, '*.{0}'.format(speech_file_type))))

        if settings.getboolean('general','transcribe'):

            # Make sure there are files to transcribe
            if len(speech_filepaths) <= 0:
                raise ValueError('There is no file with the extension "{0}"  in the folder "{1}"'.
                                 format(speech_file_type,data_folder))

            # Transcribe
            logging.debug('\n### Call the ASR engines')
            for speech_file_number, speech_filepath in enumerate(speech_filepaths):
                # Convert the speech file from FLAC/MP3/Ogg to WAV
                if speech_file_type in ['flac', 'mp3', 'ogg']:
                    from pydub import AudioSegment
                    print('speech_filepath: {0}'.format(speech_filepath))
                    sound = AudioSegment.from_file(speech_filepath, format=speech_file_type)
                    new_speech_filepath = speech_filepath[:-len(speech_file_type)-1]+'.wav'
                    sound.export(new_speech_filepath, format="wav")
                    speech_filepath = new_speech_filepath

                # Transcribe the speech file
                all_transcription_skipped = True
                for asr_system in asr_systems:
                    logging.debug('sending call to transcription')
                    transcription, transcription_skipped = transcribe.transcribe(speech_filepath,asr_system,settings,results_folder,save_transcription=True)
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
                '''
                # not sure what the purpose of this stuff is
                all_predicted_transcription_filepath = 'all_predicted_transcriptions_' + asr_system + '.txt'
                all_gold_transcription_filepath = 'all_gold_transcriptions.txt'
                all_predicted_transcription_file = codecs.open(all_predicted_transcription_filepath, 'w', settings.get('general','predicted_transcription_encoding'))
                all_gold_transcription_filepath = codecs.open(all_gold_transcription_filepath, 'w', settings.get('general','gold_transcription_encoding'))
                '''
                
                number_of_tokens_in_gold = 0
                number_of_empty_predicted_transcription_txt_files = 0
                number_of_missing_predicted_transcription_txt_files = 0
                edit_types = ['corrects', 'deletions', 'insertions', 'substitutions', 'changes']
                number_of_edits = {}

                for edit_type in edit_types:
                    number_of_edits[edit_type] = 0


                # get new results folder:
                # results_filepaths = sorted(glob.glob(os.path.join(results_folder, '*.{0}'.format(speech_file_type))))

                # store results for individual files
                result_stats_file = os.path.join(data_folder, 'stats', asr_system + '_' 'results.csv')
                logging.info('saving results to %s', result_stats_file)
                with open(result_stats_file, 'w') as f:
                    f.writelines('id,changes,corrects,substitutions,insertions,deletions,n_tokens_gold\n')

                start_time = time.time()

                threads = list()
                for speech_filepath in speech_filepaths:
                    logging.info("Main    : create and start thread %s.", speech_filepath)
                    x = threading.Thread(target=get_transcript_wer, args=(speech_filepath, results_folder, asr_system, edit_types, result_stats_file, number_of_edits, settings,))
                    threads.append(x)
                    x.start()

                for speech_filepath, thread in enumerate(threads):
                    logging.info("Main    : before joining thread %s.", speech_filepath)
                    thread.join()
                    logging.info("Main    : thread %s done", speech_filepath)


                logging.info('processed wer in %s seconds', time.time() - start_time)

def get_transcript_wer(speech_filepath, results_folder, asr_system, edit_types, result_stats_file, number_of_edits, settings):
    """
    attempt to functionalize the accuracy computation
    to allow parallel computing.
    """
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
    """
    all_predicted_transcription_file.write('{0}\n'.format(predicted_transcription))
    all_gold_transcription_filepath.write('{0}\n'.format(gold_transcription))
    """
    #print('\npredicted_transcription\t: {0}'.format(predicted_transcription))
    #print('gold_transcription\t: {0}'.format(gold_transcription))
    #TODO, handle the split just once, happens again later
    wer = metrics.wer(gold_transcription.split(' '), predicted_transcription.split(' '), False)
    #print('wer: {0}'.format(wer))

    #if len(predicted_transcription) == 0: continue

    number_of_tokens_in_gold = len(gold_transcription.split(' '))
    for edit_type in edit_types:
        number_of_edits[edit_type] += wer[edit_type]


    new_row = filename + ',' + wer_stats(wer, number_of_tokens_in_gold)
    with open(result_stats_file, 'a') as f:
        f.write(new_row + '\n')

    # return number_of_edits

def load_predicted_transcription(predicted_transcription_txt_filepath, settings):
    predicted_transcription = codecs.open(predicted_transcription_txt_filepath, 'r', settings.get('general','predicted_transcription_encoding')).read().strip()
    return predicted_transcription

def wer_stats(wer, n_tokens):
    """
    format WER stats for a csv file output.
    """
    logging.info('wer for doc %s, n_tokens %s', wer, n_tokens)
    new_row = ",".join(map(str, wer.values())) + ',' + str(n_tokens)
    logging.info('result row %s', new_row)
    return new_row

if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling
    logging.info('done')