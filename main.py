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
import glob
import torch
from torch.utils import data
from data_set import Dataset
from tqdm import tqdm as tqdm
import codecs

######################################################
params = {}
############ Data params
params['DATA_Path'] = '/mnt/Summarization/SummRunner_V2/cnn_data/finished_files/'  # './forum_data/data_V2/Parsed_Data.xml'
params['data_set_name'] = 'forum'
############ Model params
params['use_coattention'] = False
params['use_BERT'] = False
params['BERT_Model_Path'] = '../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/'
params['BERT_embedding_size'] = 768
params['BERT_layers'] = [-1]

params['embedding_size'] = 64
params['hidden_size'] = 128
params['batch_size'] = 8
params['max_num_sentences'] = 20
params['lr'] = 0.001
params['vocab_size'] = 70000
params['use_back_translation'] = False
params['back_translation_file'] = None
params['Global_max_sequence_length'] = 25
params['Global_max_num_sentences'] = 20
############ logging params
params['num_epochs'] = 50
params['start_epoch'] = 0
params['write_summarizes'] = True
params['output_dir'] = './output/'
params['save_model'] = True
params['save_model_path'] = './checkpoint/models/'
params['load_model'] = True
params['reinit_embeddings'] = True
params['load_model_path'] = './checkpoint/bilstm_model_cnn_19.pkl'
############ device
params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# params['device'] = torch.device('cpu')
###########################
params['task'] = 'Train'  ### Train, Test
params['write_post_fix'] = '0'
params['tune_postfix'] = '_tune_guf'
params['gradual_unfreezing'] = True

######################################################


def train(data_path, summRunnerModel, optimizer, criterion, parameters, epoch, gradual_unfreezing=False):

    summRunnerModel.train()

    if gradual_unfreezing is True:
        if epoch < len(summRunnerModel.layers) and epoch%3 == 0:
            current_index = int(epoch/3)
            for layer in summRunnerModel.layers: ## freeze everything
                for param in layer.parameters():
                    param.requires_grad = False
            for i in range(0, current_index + 1): ## unfreeze layers from top down to current index
                index = len(summRunnerModel.layers) - 1 - i
                for param in summRunnerModel.layers[index].parameters():
                    param.requires_grad = True
        else: ## unfreeze everything
            for layer in summRunnerModel.layers:
                for param in layer.parameters():
                    param.requires_grad = True

    epoch_loss = 0
    num_batches = 1
    for part in glob.glob(data_path):
        print('Loading training data part {}'.format(part))

        with open(part, "rb") as output_file:
            if params['use_back_translation'] is True:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part, comments_translated, posts_translated] = pickle.load(output_file)
            else:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part] = pickle.load(output_file)
                comments_translated = None
                posts_translated = None

        # data_set = Dataset(part, params['use_back_translation'])
        # data_generator = torch.utils.data.DataLoader(data_set, **parameters)
        if params['use_back_translation'] is True:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                                                                                    human_summaries_part,
                                                                                                                                                                                    sentence_str_part, params['batch_size'],
                                                                                                                                                                                    use_back_translation=params['use_back_translation'],
                                                                                                                                                                                    all_posts_translated=comments_translated,
                                                                                                                                                                                    all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches))
        else:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                             human_summaries_part,
                                                                                                                             sentence_str_part, params['batch_size'],
                                                                                                                             use_back_translation=params['use_back_translation'],
                                                                                                                             all_posts_translated=comments_translated,
                                                                                                                             all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_batches, comments_batches))
        batch_index = 1
        for post_batch, comment_batch, human_summary_batch, answer_batch, sentence_str_batch, post_translated_batch, comment_translated_batch in pbar:
            pbar.set_description("Training {}/{}, loss={}".format(batch_index, len(posts_batches), round(float(epoch_loss) / float(num_batches), 4)))
            if params['use_BERT'] is True:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_batch_BERT(comment_batch, params['BERT_layers'], params['BERT_embedding_size'])
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_batch_BERT(post_batch, params['BERT_layers'], params['BERT_embedding_size'])

            else:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_data_batch(comment_batch)
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_data_batch(post_batch)

            loss = trainer.train_batch(summRunnerModel, params['device'], post_batch, comment_batch, answer_batch,
                                       max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                                       posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                                       optimizer, criterion, params['use_BERT'])

            epoch_loss += loss
            num_batches += 1
            batch_index += 1
    print('Epoch {} Total training Loss {}'.format(epoch, round(float(epoch_loss) / float(num_batches), 4)))


def validate(data_path, summRunnerModel, optimizer, criterion, parameters, epoch, best_validation_loss):
    # print('Validating Epoch {}'.format(epoch))
    # data_path = '/checkpoint/{}_{}_*.bert.bin'.format(params['data_set_name'], 'val')
    summRunnerModel.eval()
    validation_loss = 0
    num_batches = 1

    for part in glob.glob(data_path):
        print('Loading validation data part {}'.format(part))
        with open(part, "rb") as output_file:
            if params['use_back_translation'] is True:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part, comments_translated, posts_translated] = pickle.load(output_file)
            else:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part] = pickle.load(output_file)
                comments_translated = None
                posts_translated = None

        # data_set = Dataset(part, params['use_back_translation'])
        # data_generator = torch.utils.data.DataLoader(data_set, **parameters)
        if params['use_back_translation'] is True:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                                                                                    human_summaries_part,
                                                                                                                                                                                    sentence_str_part, params['batch_size'],
                                                                                                                                                                                    use_back_translation=params['use_back_translation'],
                                                                                                                                                                                    all_posts_translated=comments_translated,
                                                                                                                                                                                    all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches))
        else:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                             human_summaries_part,
                                                                                                                             sentence_str_part, params['batch_size'],
                                                                                                                             use_back_translation=params['use_back_translation'],
                                                                                                                             all_posts_translated=comments_translated,
                                                                                                                             all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_batches, comments_batches))
        batch_index = 1
        for post_batch, comment_batch, human_summary_batch, answer_batch, sentence_str_batch, post_translated_batch, comment_translated_batch in pbar:
            pbar.set_description("Validation {}/{}, loss={}".format(batch_index, len(posts_batches), round(float(validation_loss) / float(num_batches), 4)))

            if params['use_BERT'] is True:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_batch_BERT(comment_batch, params['BERT_layers'], params['BERT_embedding_size'])
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_batch_BERT(post_batch, params['BERT_layers'], params['BERT_embedding_size'])

            else:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_data_batch(comment_batch)
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_data_batch(post_batch)

            loss = trainer.val_batch(summRunnerModel, params['device'], post_batch, comment_batch, answer_batch,
                                     max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                                     posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                                     criterion, params['use_BERT'])
            validation_loss += loss
            num_batches += 1
            batch_index += 1

    print('Epoch {} Total validation Loss {}'.format(epoch, round(float(validation_loss) / float(num_batches), 4)))
    validation_loss = float(validation_loss) / float(num_batches)

    if not os.path.exists(params['save_model_path'] + '{}{}'.format(params['data_set_name'], params['tune_postfix'])):
        os.mkdir(params['save_model_path'] + '{}{}'.format(params['data_set_name'], params['tune_postfix']))

    save_model_path = params['save_model_path'] + '{}{}/model_{}_{}_{}'.format(params['data_set_name'], params['tune_postfix'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'])

    if params['use_back_translation'] is True:
        save_model_path += '_bt'
    if params['use_coattention'] is True:
        save_model_path += '_coatt'
    if params['use_BERT'] is True:
        save_model_path += '_bert'
        save_model_path += '_{}'.format(len(params['BERT_layers']))

    if validation_loss < best_validation_loss:
        print('Best model {}, with validation loss = {}'.format(epoch, validation_loss))
        best_validation_loss = validation_loss
        best_model_epoch = epoch
        if params['save_model'] is True:
            print('Saving best model....')
            mL.save_model(summRunnerModel, optimizer, save_model_path + '_best.pkl', params)
    if params['save_model'] is True:
        print('Saving model {}'.format(epoch))
        mL.save_model(summRunnerModel, optimizer, save_model_path + '_{}.pkl'.format(epoch), params)
    return best_validation_loss


def evaluate(data_path, output_dir, summRunnerModel, parameters):
    # output_dir = params['output_dir'] + '/test_{}/'.format(epoch)
    summRunnerModel.eval()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir + '/ref/'):
        os.mkdir(output_dir + '/ref/')
    if not os.path.exists(output_dir + '/ref_abs/'):
        os.mkdir(output_dir + '/ref_abs/')
    if not os.path.exists(output_dir + '/dec/'):
        os.mkdir(output_dir + '/dec/')
    sample_index = 0
    for part in glob.glob(data_path):
        print('Loading testing data part {}'.format(part))
        with open(part, "rb") as output_file:
            if params['use_back_translation'] is True:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part, comments_translated, posts_translated] = pickle.load(output_file)
            else:
                [posts_part, comments_part, answers_part, human_summaries_part, sentence_str_part] = pickle.load(output_file)
                comments_translated = None
                posts_translated = None

        # data_set = Dataset(part, params['use_back_translation'])
        # data_generator = torch.utils.data.DataLoader(data_set, **parameters)
        if params['use_back_translation'] is True:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                                                                                    human_summaries_part,
                                                                                                                                                                                    sentence_str_part, params['batch_size'],
                                                                                                                                                                                    use_back_translation=params['use_back_translation'],
                                                                                                                                                                                    all_posts_translated=comments_translated,
                                                                                                                                                                                    all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches))
        else:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches = dL.batchify_data(posts_part, comments_part, answers_part,
                                                                                                                             human_summaries_part,
                                                                                                                             sentence_str_part, params['batch_size'],
                                                                                                                             use_back_translation=params['use_back_translation'],
                                                                                                                             all_posts_translated=comments_translated,
                                                                                                                             all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_batches, comments_batches))
        batch_index = 1
        for post_batch, comment_batch, human_summary_batch, answer_batch, sentence_str_batch, post_translated_batch, comment_translated_batch in pbar:
            pbar.set_description("Evaluating using testing data {}/{}".format(batch_index, len(posts_batches)))
            batch_index += 1
            if params['use_BERT'] is True:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_batch_BERT(comment_batch, params['BERT_layers'], params['BERT_embedding_size'])
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_batch_BERT(post_batch, params['BERT_layers'], params['BERT_embedding_size'])

            else:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = dL.pad_data_batch(comment_batch)
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = dL.pad_data_batch(post_batch)

            predicted_sentences, target_sentences, human_summaries = trainer.test_batch(summRunnerModel, params['device'], post_batch, comment_batch, answer_batch, human_summary_batch, sentence_str_batch,
                                                                                        max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                                                                                        posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths, params['use_BERT'])

            for predicted, target, human in zip(predicted_sentences, target_sentences, human_summaries):
                write_predicted = codecs.open(output_dir + '/dec/{}.dec'.format(sample_index), 'w', encoding='utf8')
                write_ref_extractive = codecs.open(output_dir + '/ref/{}.ref'.format(sample_index), 'w', encoding='utf8')
                write_ref_abstractive = codecs.open(output_dir + '/ref_abs/{}.ref'.format(sample_index), 'w', encoding='utf8')

                write_predicted.write(predicted)
                write_ref_extractive.write(target)
                write_ref_abstractive.write(human)

                write_predicted.close()
                write_ref_extractive.close()
                write_ref_abstractive.close()
                sample_index += 1


def main():
    print(params)
    '''
    '''
    ''' Initialize model and optimizer'''
    file_name = './checkpoint/data/@dataset@/@dataset@_{}.pickle'.replace('@dataset@', params['data_set_name'])
    if params['back_translation_file'] is True:
        file_name = './checkpoint/data/@dataset@/@dataset@_{}_bt.pickle'.replace('@dataset@', params['data_set_name'])

    if not os.path.exists(file_name.format('vocab')):
        print('Vocab file doesnot exist make sure you preprocessed the data before running training')
        print('Exiting......')
        exit()

    with open(file_name.format('vocab'), "rb") as output_file:
        [word2id_dictionary, id2word_dictionary] = pickle.load(output_file)
    vocab_size = len(word2id_dictionary)
    max_number_sentences = params['Global_max_num_sentences'] + 1  # max([max(train_max_sentences), max(val_max_sentences), max(test_max_sentences)]) + 1

    params['max_num_sentences'] = max_number_sentences
    params['vocab_size'] = vocab_size

    summRunnerModel = mL.init_model(params, vocab_size)
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(summRunnerModel.parameters(), lr=params['lr'])  # 1e-3)

    if params['load_model'] is True:
        print('Loading Model from {}'.format(params['load_model_path']))
        summRunnerModel, optimizer = mL.load_model(optimizer=optimizer, path=params['load_model_path'], device=params['device'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = params['lr']

    if params['reinit_embeddings'] is True:
        print('Reinitializing model embeddings....')
        summRunnerModel = mL.reinit_embedding_layer(summRunnerModel, vocab_size, params['embedding_size'], params['max_num_sentences'])
        if params['device'].type == 'cuda':
            summRunnerModel.cuda()

    # Parameters
    parameters = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 4}

    best_validation_loss = math.inf
    best_model_epoch = 0
    if params['task'] == 'Train':
        for epoch in range(params['start_epoch'], params['num_epochs']):
            print('Training Epoch {}'.format(epoch))
            ########################### Train ################################################
            if params['use_BERT'] is True:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_{}_*.bin.bert'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), 'train')
            else:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_*.bin'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], 'train')
            parts = glob.glob(data_path)
            if len(parts) == 0:
                print('No training data found, make sure you preprocessed the data first')
                exit()
            else:
                train(data_path, summRunnerModel, optimizer, criterion, parameters, epoch, gradual_unfreezing=params['gradual_unfreezing'])

            ########################### Validation ################################################
            print('Validating Epoch {}'.format(epoch))
            if params['use_BERT'] is True:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_{}_*.bin.bert'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), 'val')
            else:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_*.bin'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], 'val')

            parts = glob.glob(data_path)
            if len(parts) == 0:
                print('No validation data found, make sure you preprocessed the data first')
            else:
                best_validation_loss = validate(data_path, summRunnerModel, optimizer, criterion, parameters, epoch, best_validation_loss)

            ########################### Test ################################################
            print('Evaluating using testing data Epoch {}'.format(epoch))
            if params['use_BERT'] is True:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_{}_*.bin.bert'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), 'test')
            else:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_*.bin'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], 'test')
            summRunnerModel.eval()

            output_dir = params['output_dir'] + '/{}_{}_{}{}'.format(params['Global_max_num_sentences'], params['Global_max_sequence_length'], params['data_set_name'], params['tune_postfix'])
            if params['use_BERT'] is True:
                output_dir += '{}_bert'.format(len(params['BERT_layers']))
            if params['use_coattention'] is True:
                output_dir += '_coatt'
            output_dir += '/'

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = output_dir + '/test_{}/'.format(epoch)

            parts = glob.glob(data_path)
            if len(parts) == 0:
                print('No testing data found, make sure you preprocessed the data first')
            else:
                evaluate(data_path, output_dir, summRunnerModel, parameters)

            #############################################################################################################################################################
            print('Evaluating using validation data Epoch {}'.format(epoch))
            if params['use_BERT'] is True:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_{}_*.bin.bert'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), 'val')
            else:
                data_path = './checkpoint/data/{}/{}_{}_{}_{}_*.bin'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], 'val')
            summRunnerModel.eval()

            output_dir = params['output_dir'] + '/{}_{}_{}{}'.format(params['Global_max_num_sentences'], params['Global_max_sequence_length'], params['data_set_name'], params['tune_postfix'])
            if params['use_BERT'] is True:
                output_dir += '{}_bert'.format(len(params['BERT_layers']))
            if params['use_coattention'] is True:
                output_dir += '_coatt'
            output_dir += '/'

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = output_dir + '/val_{}/'.format(epoch)

            parts = glob.glob(data_path)
            if len(parts) == 0:
                print('No validation data found, make sure you preprocessed the data first')
            else:
                evaluate(data_path, output_dir, summRunnerModel, parameters)

    elif params['task'] == 'Test':

        ########################### Test ################################################
        print('Evaluating using training data')
        if params['use_BERT'] is True:
            data_path = './checkpoint/data/{}/{}_{}_{}_{}_{}_*.bin.bert'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], len(params['BERT_layers']), 'test')
        else:
            data_path = './checkpoint/data/{}/{}_{}_{}_{}_*.bin'.format(params['data_set_name'], params['data_set_name'], params['Global_max_num_sentences'], params['Global_max_sequence_length'], 'test')
        summRunnerModel.eval()

        output_dir = params['output_dir'] + '/{}_{}_{}{}'.format(params['Global_max_num_sentences'], params['Global_max_sequence_length'], params['data_set_name'], params['tune_postfix'])
        if params['use_BERT'] is True:
            output_dir += '{}_bert'.format(len(params['BERT_layers']))
        if params['use_coattention'] is True:
            output_dir += '_coatt'
        output_dir += '/'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = output_dir + '/test/'

        parts = glob.glob(data_path)
        if len(parts) == 0:
            print('No testing data found, make sure you preprocessed the data first')
        else:
            evaluate(data_path, output_dir, summRunnerModel, parameters)


if __name__ == '__main__':
    # for dsn in ['forum']:  # , 'github']:
    #     for i in [20, 30, 40, 50]:
    #         for j in [25, 35, 55, 75]:
    #             for coatt in [True, False]:
    #                 for ub in [True, False]:
    #                     if dsn == 'github':
    #                         params['DATA_Path'] = './github_data/issues_v2_combined.xml'
    #                     elif dsn == 'cnn':
    #                         params['DATA_Path'] = './cnn_data/finished_files/'
    #                     elif dsn == 'forum':
    #                         params['DATA_Path'] = './forum_data/data_V2/Parsed_Data.xml'
    #
    #                     params['data_set_name'] = dsn
    #                     params['use_BERT'] = ub
    #                     params['Global_max_sequence_length'] = j
    #                     params['Global_max_num_sentences'] = i
    #                     params['max_num_sentences'] = i
    #                     params['use_coattention'] = coatt
    #                     params['BERT_layers'] = [-1, -2]
    #
    #                     print('{} {} {} {}'.format(i, j, ub, dsn))
    main()
