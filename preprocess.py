# import HelpingFunctions
# from model import EncoderBiRNN
# import constants
import torch.nn as nn
import torch
import data_loader as dL
import model_loader as mL
import random
import math
import trainer as trainer
from time import sleep
import pickle
import os
import HelpingFunctions as HelpingFunctions

######################################################
params = {}
############ Data params
params['DATA_Path'] = './cnn_data/finished_files/'#'./github_data/issues_v2_combined.xml'#'./cnn_data/finished_files/' #'./forum_data/data_V2/Parsed_Data.xml'
params['data_set_name'] = 'cnn'
############ Model params
params['use_BERT'] = False
params['BERT_Model_Path'] = '../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/'
params['BERT_embedding_size'] = 768
params['BERT_layers'] = [-1]

params['vocab_size'] = 70000
params['use_back_translation'] = False
params['back_translation_file'] = None
params['Global_max_sequence_length'] = 25
params['Global_max_num_sentences'] = 20
params['use_external_vocab'] = False
params['external_vocab_file'] = './checkpoint/forum_vocab.pickle'
params['encoding_batch_size'] = 16
params['data_split_size'] = 15000
params['extract_keywords'] = True
############ device
params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
######################################################

def read_data():
    '''
    '''
    ''' 
    Load data
    '''
    file_name = './checkpoint/data/{}/{}_@dpart@.pickle'.format(params['data_set_name'], params['data_set_name'])
    if params['back_translation_file'] is True:
        file_name = './checkpoint/{}/{}_@dpart@_bt.pickle'.format(params['data_set_name'], params['data_set_name'])
    if os.path.exists(file_name.replace('@dpart@', 'vocab')) and os.path.exists(file_name.replace('@dpart@', 'test')) and os.path.exists(file_name.replace('@dpart@', 'train')) and os.path.exists(file_name.replace('@dpart@', 'val')):
        print('Data exists will not parse')
        return
    
    if params['data_set_name'] == 'cnn':
        print('Loading CNN data')
        train_data, val_data, test_data, word2id_dictionary, id2word_dictionary = dL.read_cnn_dm_data(params['DATA_Path'],
                                                                                                      limit_vocab=params['vocab_size'],
                                                                                                      use_back_translation=params['use_back_translation'],
                                                                                                      back_translation_file=params['back_translation_file'])
        
    elif params['data_set_name'] == 'github':
        train_data, val_data, test_data, word2id_dictionary, id2word_dictionary = dL.read_github_data(params['DATA_Path'],
                                                                    use_back_translation=params['use_back_translation'],
                                                                    back_translation_file=params['back_translation_file'])
        
    else:
        print('Loading {} data'.format(params['data_set_name']))
        train_data, val_data, test_data, word2id_dictionary, id2word_dictionary = dL.read_data(params['DATA_Path'],
                                                                    use_back_translation=params['use_back_translation'],
                                                                    back_translation_file=params['back_translation_file'])


    print('Saving data...')
    with open(file_name.replace('@dpart@', 'train'), "wb") as output_file:
        pickle.dump(train_data, output_file)
    with open(file_name.replace('@dpart@', 'val'), "wb") as output_file:
        pickle.dump(val_data, output_file)
    with open(file_name.replace('@dpart@', 'test'), "wb") as output_file:
        pickle.dump(test_data, output_file)
    with open(file_name.replace('@dpart@', 'vocab'), "wb") as output_file:
        pickle.dump([word2id_dictionary, id2word_dictionary], output_file)

    del train_data, val_data, test_data, word2id_dictionary, id2word_dictionary


def tokenize_data(data, data_part):
    if params['use_back_translation'] is True:
        file_path = './checkpoint/data/{}/{}_{}_bt.tkz.bin'.format(params['data_set_name'], params['data_set_name'], data_part)
    else:
        file_path = './checkpoint/data/{}/{}_{}.tkz.bin'.format(params['data_set_name'], params['data_set_name'], data_part)

    all_posts_translated = None
    all_comments_translated = None

    all_post_keywords = None
    all_comment_keywords = None

    if os.path.exists(file_path):
        print('Loading saved tokenized data from {}'.format(file_path))
        with open(file_path, "rb") as output_file:
            if params['use_back_translation'] is True:
                [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated] = pickle.load(output_file)
            elif params['extract_keywords'] is True:
                [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_post_keywords, all_comment_keywords] = pickle.load(output_file)
            else:
                [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str] = pickle.load(output_file)
    else:
        if params['use_back_translation'] is True:
            all_posts, all_comments, all_answers, all_human_summaries, all_posts_translated, all_comments_translated = dL.tokenize_data(data, use_back_translation=True)
        elif params['extract_keywords'] is True:
            all_posts, all_comments, all_answers, all_human_summaries, all_post_keywords, all_comment_keywords = dL.tokenize_data(data, use_back_translation=False, extract_keywords=True)
        else:
            all_posts, all_comments, all_answers, all_human_summaries = dL.tokenize_data(data)

        all_sentence_str = []
        for index, comment in enumerate(all_comments):
            all_sentence_str.append(all_comments[index])

        if params['use_back_translation'] is True:
            save_list = [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated]
        elif params['extract_keywords'] is True:
            save_list = [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_post_keywords, all_comment_keywords]
        else:
            save_list = [all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str]

        print('Saving tokenized data')
        with open(file_path, "wb") as output_file:
            pickle.dump(save_list, output_file)

    return all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated, all_post_keywords, all_comment_keywords


def encode_data(all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated, word2id_dictionary, data_part, all_post_keywords=None, all_comment_keywords=None):
    use_BERT = params['use_BERT']
    encoding_batch_size = params['encoding_batch_size']
    split_size = params['data_split_size']

    if len(all_comments) < split_size:
        num_parts = 1
    else:
        num_parts = int(len(all_comments) / split_size)

    for part_num in range(num_parts):
        if part_num == num_parts - 1:
            start = part_num * split_size
            end = len(all_comments)
        else:
            start = part_num * split_size
            end = (part_num + 1) * split_size

        print('Encoded {} data...... part {}/{}'.format(data_part, part_num + 1, num_parts))

        all_posts_part = all_posts[start: end]
        all_comments_part = all_comments[start: end]
        all_answers_part = all_answers[start: end]
        all_human_summaries_part = all_human_summaries[start: end]
        all_sentence_str_part = all_sentence_str[start: end]
        if params['use_back_translation'] is True:
            all_posts_translated_part = all_posts_translated[start: end]
            all_comments_translated_part = all_comments_translated[start: end]
        if params['extract_keywords'] is True and all_post_keywords is not None and all_comment_keywords is not None:
            all_post_keywords_part = all_post_keywords[start: end]
            all_comment_keywords_part = all_comment_keywords[start: end]

        if use_BERT is True:
            all_comments_part = dL.encode_data_BERT(all_comments_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)
            all_posts_part = dL.encode_data_BERT(all_posts_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)
            if params['use_back_translation'] is True:
                all_comments_translated_part = dL.encode_data_BERT(all_comments_translated_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)
                all_posts_translated_part = dL.encode_data_BERT(all_posts_translated_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)
            if params['extract_keywords'] is True:
                all_comment_keywords_part = dL.encode_data_BERT(all_comment_keywords_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)
                all_post_keywords_part = dL.encode_data_BERT(all_post_keywords_part, params['BERT_Model_Path'], params['device'], params['BERT_layers'], encoding_batch_size)

        else:
            all_comments_part = dL.encode_data(all_comments_part, word2id_dictionary)
            all_posts_part = dL.encode_data(all_posts_part, word2id_dictionary)
            if params['use_back_translation'] is True:
                all_comments_translated_part = dL.encode_data(all_comments_translated_part, word2id_dictionary)
                all_posts_translated_part = dL.encode_data(all_posts_translated_part, word2id_dictionary)
            if params['extract_keywords'] is True:
                all_comment_keywords_part = dL.encode_data(all_comment_keywords_part, word2id_dictionary)
                all_post_keywords_part = dL.encode_data(all_post_keywords_part, word2id_dictionary)

        ####### Saving Data
        if params['use_back_translation'] is True:
            save_list = [all_posts_part, all_comments_part, all_answers_part, all_human_summaries_part, all_sentence_str_part, all_posts_translated_part, all_comments_translated_part]
        elif params['extract_keywords'] is True:
            save_list = [all_posts_part, all_comments_part, all_answers_part, all_human_summaries_part, all_sentence_str_part, all_post_keywords_part, all_comment_keywords_part]
        else:
            save_list = [all_posts_part, all_comments_part, all_answers_part, all_human_summaries_part, all_sentence_str_part]

        print('Saving encoded data')

        if not os.path.exists('./checkpoint/data/{}/'.format(params['data_set_name'])):
            os.mkdir('./checkpoint/data/{}/'.format(params['data_set_name']))
        parent_dir = './checkpoint/data/{}/'.format(params['data_set_name'])

        if use_BERT is True:
            if params['use_external_vocab'] is True:
                file_path = parent_dir + '{}_externalvocab_{}_{}_{}_{}_{}.bin.bert'.format(params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), data_part, part_num)
            else:
                file_path = parent_dir + '{}_{}_{}_{}_{}_{}.bin.bert'.format(params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), data_part, part_num)
        else:
            if params['use_external_vocab'] is True:
                file_path = parent_dir + '{}_externalvocab_{}_{}_{}_{}.bin'.format(params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], data_part, part_num)
            else:
                file_path = parent_dir + '{}_{}_{}_{}_{}.bin'.format(params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], data_part, part_num)
        with open(file_path, "wb") as output_file:
            pickle.dump(save_list, output_file)

        del all_posts_part, all_comments_part, all_answers_part, all_human_summaries_part, all_sentence_str_part
        del save_list
        for j in range(start, end):
            if params['use_back_translation'] is True:
                all_posts_translated[j] = None
                all_comments_translated[j] = None
            if params['extract_keywords'] is True:
                all_post_keywords[j] = None
                all_comment_keywords[j] = None
            all_posts[j] = None
            all_comments[j] = None
            all_human_summaries[j] = None
            all_answers[j] = None
            all_sentence_str[j] = None
            

def main():
    '''
    '''
    ''' 
    Load data
    '''

    if not os.path.exists('./checkpoint/data/{}/'.format(params['data_set_name'])):
        os.mkdir('./checkpoint/data/{}/'.format(params['data_set_name']))

    read_data()

    file_name = './checkpoint/data/{}/{}_@dpart@.pickle'.format(params['data_set_name'], params['data_set_name'])
    if params['back_translation_file'] is True:
        file_name = './checkpoint/data/{}/{}_@dpart@_bt.pickle'.format(params['data_set_name'], params['data_set_name'])

    if params['use_external_vocab'] is True:
        with open(params['external_vocab_file'], "rb") as output_file:
            [word2id_dictionary, id2word_dictionary] = pickle.load(output_file)
        with open(file_name.replace('@dpart@', 'vocab').replace(params['data_set_name']+'_vocab', params['data_set_name']+'_externalvocab_vocab'), "wb") as output_file:
             pickle.dump([word2id_dictionary, id2word_dictionary], output_file)
    else:
        with open(file_name.replace('@dpart@', 'vocab'), "rb") as output_file:
            [word2id_dictionary, id2word_dictionary] = pickle.load(output_file)

    for data_part in ['test', 'train', 'val']:
        print('Processing {}'.format(data_part))
        with open(file_name.replace('@dpart@', data_part), "rb") as output_file:
            data = pickle.load(output_file)
        all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated, all_post_keywords, all_comment_keywords = tokenize_data(data, data_part)
        del data

        '''
        Reduce data size to max sentences and max sequence length
        '''

        max_num_sentences = params['Global_max_num_sentences']
        max_sentence_length = params['Global_max_sequence_length']

        if max_num_sentences is not None:
            for index, comment in enumerate(all_comments):
                all_comments[index] = all_comments[index][:max_num_sentences]
                all_sentence_str[index] = all_sentence_str[index][:max_num_sentences]

            for index, answer in enumerate(all_answers):
                all_answers[index] = all_answers[index][:max_num_sentences]

            if params['use_back_translation'] is True:
                for index, comment in enumerate(all_comments_translated):
                    all_comments_translated[index] = all_comments_translated[index][:max_num_sentences]
            if params['extract_keywords'] is True:
                for index, comment in enumerate(all_comment_keywords):
                    all_comment_keywords[index] = all_comment_keywords[index][:max_num_sentences]

        if max_sentence_length is not None:
            for index, comment in enumerate(all_comments):
                for index_2, sent in enumerate(comment):
                    all_comments[index][index_2] = all_comments[index][index_2][:max_sentence_length]

            if params['use_back_translation'] is True:
                for index, comment in enumerate(all_comments_translated):
                    for index_2, sent in enumerate(comment):
                        all_comments_translated[index][index_2] = all_comments_translated[index][index_2][:max_sentence_length]

            if params['extract_keywords'] is True:
                for index, comment in enumerate(all_comment_keywords):
                    for index_2, sent in enumerate(comment):
                        all_comment_keywords[index][index_2] = all_comment_keywords[index][index_2][:max_sentence_length]

        encode_data(all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated, word2id_dictionary, data_part, all_post_keywords, all_comment_keywords)


if __name__ == '__main__':
    ######################################################
    for dsn in ['forum_keywords']:#, 'github', 'cnn']:
        for i in [30]:#[20, 30, 40, 50]:
            for j in [75]:#[25, 35, 55, 75]:
                for ub in [False, True]:
                    if dsn == 'github':
                        params['DATA_Path'] = './github_data/issues_v2_combined.xml'
                    elif dsn == 'cnn':
                        params['DATA_Path'] = './cnn_data/finished_files/'
                    elif dsn == 'forum' or dsn == 'forum_keywords':
                        params['DATA_Path'] = './forum_data/data_V2/Parsed_Data.xml'

                    params['data_set_name'] = dsn
                    params['use_BERT'] = ub
                    params['Global_max_sequence_length'] = j
                    params['Global_max_num_sentences'] = i
                    # try:
                    if ub is True:
                        for blyaers in [[-1, -2]]:
                            params['BERT_layers'] = blyaers
                            print('{} {} {} {} {}'.format(i, j, ub, dsn, blyaers))
                            main()
                    else:
                        print('{} {} {} {}'.format(i, j, ub, dsn))
                        main()
                    # except Exception as e:
                    #     print(e)
                    #     continue



