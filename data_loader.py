import HelpingFunctions as hF
import constants as constants
from Data.reply_thread import ReplyThread
from lxml import etree as ET
import os
import pickle
from tqdm import tqdm as tqdm_notebook
import torch


def read_github_data(data_file_path, use_back_translation=False, back_translation_file=None):
    train_data = []
    val_data = []
    test_data = []

    word2id_dictionary = {}
    id2word_dictionary = {}

    tree = ET.parse(data_file_path)
    root = tree.getroot()
    words = []

    train_ratio = 0
    val_ratio = 0
    test_ratio = 1

    all_data_length = len(root)
    num_train = int(train_ratio * all_data_length)
    num_val = int(val_ratio * all_data_length)
    num_test = all_data_length - num_train - num_val#  test_ratio * all_data_length

    for index, post_item in tqdm_notebook(enumerate(root)):
        post_id = post_item.tag
        title = [item.text for item in post_item.findall('Title')][0]
        words += hF.tokenize_text(title)

        owner_id = [item.text for item in post_item.findall('Owner')][0]

        initial_comment = [item.text for item in post_item.findall('Body')][0].replace('\r', '').split('\n')
        initial_comment = [x.strip() for x in initial_comment if x.strip() != '']
        for sentence in initial_comment:
            words += hF.tokenize_text(sentence)

        summary_1 = initial_comment[0]
        summary_2 = initial_comment[0]
        selected_sentences = [initial_comment[0]]

        rthread = ReplyThread(post_id, title, initial_comment, owner_id, summary_1, summary_2, selected_sentences)

        ################ add initial post to comments as well
        rthread.add_reply(initial_comment)
        #####################################################
        for comment_item in post_item.findall('Comment'):

            comment_body = [item.text for item in comment_item.findall('Body')][0]
            if comment_body is None:
                continue
            comment_body = [x.strip() for x in comment_body.replace('\r', '').split('\n') if x.strip() != '']

            for sentence in comment_body:
                words += hF.tokenize_text(sentence)
            rthread.add_reply(comment_body)

        if len(rthread.reply_sentences) > 1:
            if index < num_train:
                train_data.append(rthread)
            elif num_train <= index < num_train + num_val:
                val_data.append(rthread)
            else:
                test_data.append(rthread)
        # else:
        #     print('empty thread')

    '''
    Read backtranslation data and fill it in the array 
    '''
    print('Adding back translation info...')
    if use_back_translation is True and back_translation_file is not None:
        reader = open(back_translation_file, 'r')
        id = ''
        comments = []
        initial_comment = []
        selected_sentences = []

        part_ = None
        for line in reader:
            line = line.replace('\n','').replace('\r','').strip()
            if line == '':
                continue

            if '@START_Art@' in line:
                if id != '':
                    ''' Add previous one'''
                    for data in [train_data, val_data, test_data]:
                        for elem in data:
                            if elem.post_id == id:
                                if len(initial_comment) > 1:
                                    initial_comment = initial_comment[1:]
                                elem.initial_post_translated = initial_comment
                                elem.reply_sentences_translated = comments
                                elem.selected_sentences_translated = selected_sentences

                                words += hF.tokenize_text(' '.join(initial_comment))
                                words += hF.tokenize_text(' '.join(comments))
                                break

                id = line.split(' ')[1].replace('@', '').strip()
                comments = []
                initial_comment = []
                selected_sentences = []
                part_ = 'init'
            elif '@COMMENTS@' in line:
                part_ = 'comments'
            elif '@HIGHLIGHT@' in line:
                part_ = 'summary'
            else:
                if part_ == 'comments':
                    comments.append(line)
                elif part_ == 'summary':
                    selected_sentences.append(line)
                elif part_ == 'init':
                    initial_comment.append(line)

        if id != '':
            ''' Add Last one'''
            for data in [train_data, val_data, test_data]:
                for elem in data:
                    if elem.post_id == id:
                        if len(initial_comment) > 1:
                            initial_comment = initial_comment[1:]
                        elem.initial_post_translated = initial_comment
                        elem.reply_sentences_translated = comments
                        elem.selected_sentences_translated = selected_sentences
                        break
    '''
    Fill Dictionary
    '''
    ########################################## Initialize_Dictionary############################
    word2id_dictionary[constants.pad_token] = constants.pad_index
    word2id_dictionary[constants.oov_token] = constants.oov_index
    word2id_dictionary[constants.SOS_token] = constants.SOS_index
    word2id_dictionary[constants.EOS_token] = constants.EOS_index

    id2word_dictionary[constants.pad_index] = constants.pad_token
    id2word_dictionary[constants.oov_index] = constants.oov_token
    id2word_dictionary[constants.SOS_index] = constants.SOS_token
    id2word_dictionary[constants.EOS_index] = constants.EOS_token
    ############################################################################################
    for word in words:
        if word.strip() not in word2id_dictionary:
            index = len(word2id_dictionary.keys())
            word2id_dictionary[word.strip()] = index
            id2word_dictionary[index] = word.strip()

    # print('Saving data...')
    # save_list = [data, word2id_dictionary, id2word_dictionary]
    # with open(file_name, "wb") as output_file:
    #     pickle.dump(save_list, output_file)

    return train_data, val_data, test_data, word2id_dictionary, id2word_dictionary


def read_data(data_file_path, use_back_translation=False, back_translation_file=None):
    # file_name = './checkpoint/forum_data_dic.pickle'
    # if use_back_translation is True:
    #     file_name = './checkpoint/forum_data_dic_bt.pickle'
    #
    # if os.path.exists(file_name):
    #     with open(file_name, "rb") as output_file:
    #         [data, word2id_dictionary, id2word_dictionary] = pickle.load(output_file)
    #         return data, word2id_dictionary, id2word_dictionary

    train_data = []
    val_data = []
    test_data = []

    word2id_dictionary = {}
    id2word_dictionary = {}

    tree = ET.parse(data_file_path)
    root = tree.getroot()
    words = []

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    all_data_length = len(root)
    num_train = int(train_ratio * all_data_length)
    num_val = int(val_ratio * all_data_length)
    num_test = all_data_length - num_train - num_val#  test_ratio * all_data_length

    for index, post_item in tqdm_notebook(enumerate(root)):
        post_id = post_item.tag
        title = [item.text for item in post_item.findall('Title')][0]
        words += hF.tokenize_text(title)

        owner_id = [item.text for item in post_item.findall('Owner')][0]

        initial_comment = [item.text for item in post_item.findall('Body')][0].split('\n')
        for sentence in initial_comment:
            words += hF.tokenize_text(sentence)

        summary_1 = [item.text for item in post_item.findall('Summary_1')][0]
        words += hF.tokenize_text(summary_1)
        summary_2 = [item.text for item in post_item.findall('Summary_2')][0]
        words += hF.tokenize_text(summary_1)

        selected_sentences = [item.text for item in post_item.findall('selected_sentences')][0].split('\n')
        for sentence in selected_sentences:
            words += hF.tokenize_text(sentence)

        rthread = ReplyThread(post_id, title, initial_comment, owner_id, summary_1, summary_2, selected_sentences)

        ################ add initial post to comments as well
        rthread.add_reply(initial_comment)
        #####################################################
        for comment_item in post_item.findall('Comment'):
            comment_body = [item.text for item in comment_item.findall('Body')][0].split('\n')
            for sentence in comment_body:
                words += hF.tokenize_text(sentence)
            rthread.add_reply(comment_body)

        if index < num_train:
            train_data.append(rthread)
        elif num_train <= index < num_train + num_val:
            val_data.append(rthread)
        else:
            test_data.append(rthread)

    '''
    Read backtranslation data and fill it in the array 
    '''
    print('Adding back translation info...')
    if use_back_translation is True and back_translation_file is not None:
        reader = open(back_translation_file, 'r')
        id = ''
        comments = []
        initial_comment = []
        selected_sentences = []

        part_ = None
        for line in reader:
            line = line.replace('\n','').replace('\r','').strip()
            if line == '':
                continue

            if '@START_Art@' in line:
                if id != '':
                    ''' Add previous one'''
                    for data in [train_data, val_data, test_data]:
                        for elem in data:
                            if elem.post_id == id:
                                if len(initial_comment) > 1:
                                    initial_comment = initial_comment[1:]
                                elem.initial_post_translated = initial_comment
                                elem.reply_sentences_translated = comments
                                elem.selected_sentences_translated = selected_sentences

                                words += hF.tokenize_text(' '.join(initial_comment))
                                words += hF.tokenize_text(' '.join(comments))
                                break

                id = line.split(' ')[1].replace('@', '').strip()
                comments = []
                initial_comment = []
                selected_sentences = []
                part_ = 'init'
            elif '@COMMENTS@' in line:
                part_ = 'comments'
            elif '@HIGHLIGHT@' in line:
                part_ = 'summary'
            else:
                if part_ == 'comments':
                    comments.append(line)
                elif part_ == 'summary':
                    selected_sentences.append(line)
                elif part_ == 'init':
                    initial_comment.append(line)

        if id != '':
            ''' Add Last one'''
            for data in [train_data, val_data, test_data]:
                for elem in data:
                    if elem.post_id == id:
                        if len(initial_comment) > 1:
                            initial_comment = initial_comment[1:]
                        elem.initial_post_translated = initial_comment
                        elem.reply_sentences_translated = comments
                        elem.selected_sentences_translated = selected_sentences
                        break
    '''
    Fill Dictionary
    '''
    ########################################## Initialize_Dictionary############################
    word2id_dictionary[constants.pad_token] = constants.pad_index
    word2id_dictionary[constants.oov_token] = constants.oov_index
    word2id_dictionary[constants.SOS_token] = constants.SOS_index
    word2id_dictionary[constants.EOS_token] = constants.EOS_index

    id2word_dictionary[constants.pad_index] = constants.pad_token
    id2word_dictionary[constants.oov_index] = constants.oov_token
    id2word_dictionary[constants.SOS_index] = constants.SOS_token
    id2word_dictionary[constants.EOS_index] = constants.EOS_token
    ############################################################################################
    for word in words:
        if word.strip() not in word2id_dictionary:
            index = len(word2id_dictionary.keys())
            word2id_dictionary[word.strip()] = index
            id2word_dictionary[index] = word.strip()

    # print('Saving data...')
    # save_list = [data, word2id_dictionary, id2word_dictionary]
    # with open(file_name, "wb") as output_file:
    #     pickle.dump(save_list, output_file)

    return train_data, val_data, test_data, word2id_dictionary, id2word_dictionary


def read_cnn_dm_data(data_parent_dir, limit_vocab=70000, use_back_translation=False, back_translation_file=None):
    import json

    # file_name = './checkpoint/cnn_data_dic.pickle'
    # if use_back_translation is True:
    #     file_name = './checkpoint/cnn_data_dic_bt.pickle'
    #
    # if os.path.exists(file_name):
    #     with open(file_name, "rb") as output_file:
    #         [data, word2id_dictionary, id2word_dictionary] = pickle.load(output_file)
    #         return data, word2id_dictionary, id2word_dictionary

    train_data = []
    val_data = []
    test_data = []
    word2id_dictionary = {}
    id2word_dictionary = {}
    ########################################## Initialize_Dictionary############################
    word2id_dictionary[constants.pad_token] = constants.pad_index
    word2id_dictionary[constants.oov_token] = constants.oov_index
    word2id_dictionary[constants.SOS_token] = constants.SOS_index
    word2id_dictionary[constants.EOS_token] = constants.EOS_index

    id2word_dictionary[constants.pad_index] = constants.pad_token
    id2word_dictionary[constants.oov_index] = constants.oov_token
    id2word_dictionary[constants.SOS_index] = constants.SOS_token
    id2word_dictionary[constants.EOS_index] = constants.EOS_token
    ############################################################################################

    word_frequency = {}
    test_count = 0
    train_count = 0
    val_count = 0
    for data_dir in ['test', 'train', 'val']:
        words = []
        file_names = os.listdir(data_parent_dir + '//' + data_dir)
        for fname in tqdm_notebook(file_names):
            with open(data_parent_dir + '//' + data_dir + '//' + fname) as json_file:
                data = json.load(json_file)
                article = [x.strip() for x in data['article']]
                if len(article) < 2:
                    continue
                human_summary = ' '.join([x.strip() for x in data['abstract']])
                summary = [article[x] for x in data['extracted']]

                words += hF.tokenize_text(' '.join(article))
                words += hF.tokenize_text(human_summary)

                '''
                Filter data
                '''
                initial_comment = article[0]
                if initial_comment.strip() == '' or len(initial_comment.strip()) <= 2:
                    continue
                initial_comment = [initial_comment]
                article = [x for x in article if x.strip() != '' and len(x) > 2]
                if len(article) <= 1:
                    continue
                '''
                end Data filtering
                '''
                rThread = ReplyThread(fname, fname, initial_comment, '', human_summary, '', summary)
                rThread.add_reply(article)

                if data_dir == 'train':
                    train_data.append(rThread)
                    train_count += 1
                elif data_dir == 'test':
                    test_data.append(rThread)
                    test_count += 1
                elif data_dir == 'val':
                    val_data.append(rThread)
                    val_count += 1

        for word in words:
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        del words

    '''
    Read back translation data and fill it in the array 
    '''
    print('Adding back translation info...')
    words = []
    if use_back_translation is True and back_translation_file is not None:
        reader = open(back_translation_file, 'r')
        id = ''
        comments = []
        initial_comment = []
        selected_sentences = []

        part_ = None
        for line in reader:
            line = line.replace('\n','').replace('\r','').strip()
            if line == '':
                continue

            if '@START_Art@' in line:
                if id != '':
                    ''' Add previous one'''
                    for data in [train_data, val_data, test_data]:
                        for elem in data:
                            if elem.post_id == id:
                                if len(initial_comment) > 1:
                                    initial_comment = initial_comment[1:]
                                elem.initial_post_translated = initial_comment
                                elem.reply_sentences_translated = comments
                                elem.selected_sentences_translated = selected_sentences

                                words += hF.tokenize_text(' '.join(initial_comment))
                                words += hF.tokenize_text(' '.join(comments))
                                break

                id = line.split(' ')[1].replace('@', '').strip()
                comments = []
                initial_comment = []
                selected_sentences = []
                part_ = 'init'
            if '@COMMENTS@' in line:
                part_ = 'comments'
            elif '@HIGHLIGHT@' in line:
                part_ = 'summary'
            else:
                if part_ == 'comments':
                    comments.append(line)
                elif part_ == 'summary':
                    selected_sentences.append(line)
                elif part_ == 'init':
                    initial_comment.append(line)

        if id != '':
            ''' Add Last one'''
            for data in [train_data, val_data, test_data]:
                for elem in data:
                    if elem.post_id == id:
                        if len(initial_comment) > 1:
                            initial_comment = initial_comment[1:]
                        elem.initial_post_translated = initial_comment
                        elem.reply_sentences_translated = comments
                        elem.selected_sentences_translated = selected_sentences
                        break
    for word in words:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
    del words
    '''
    End back translation part
    '''
    '''
    '''
    import operator
    word_frequency = sorted(word_frequency.items(), key=operator.itemgetter(1), reverse=True)
    word_frequency = word_frequency[:limit_vocab]
    for word_count in word_frequency:
        word = word_count[0]
        if word.strip() not in word2id_dictionary:
            index = len(word2id_dictionary.keys())
            word2id_dictionary[word.strip()] = index
            id2word_dictionary[index] = word.strip()
    del word_frequency

    return train_data, val_data, test_data, word2id_dictionary, id2word_dictionary


def tokenize_data(data, use_back_translation=False):#, max_num_sentences=None, max_sentence_length=None):
    all_comments = []
    all_posts = []
    all_answers = []
    all_human_summaries = []
    
    all_comments_translated = []
    all_posts_translated = []

    print('Tokenizing Data...')
    for i in tqdm_notebook(range(0, len(data))):
        post = [x.replace('\n','').replace('\r','').strip() for x in data[i].initial_post]
        comments = [x.replace('\n','').replace('\r','').strip() for x in data[i].reply_sentences]
        selected_sentences = [x.replace('\n','').replace('\r','').strip() for x in data[i].selected_sentences]

        answers = [1 if x in selected_sentences else 0 for x in comments]
        post = [hF.tokenize_text(x) for x in post]

        comments = [hF.tokenize_text(x) for x in comments]
        human_summary = data[i].summary_1

        if use_back_translation is True:
            post_translated = [x.replace('\n','').replace('\r','').strip().split(' ') for x in data[i].initial_post_translated]
            comments_translated = [x.replace('\n','').replace('\r','').strip() for x in data[i].reply_sentences_translated]


            post_translated = [x.split(' ') for x in post_translated]
            all_comments_translated.append(comments_translated)
            all_posts_translated.append(post_translated)

        all_answers.append(answers)
        all_comments.append(comments)
        all_posts.append(post)
        all_human_summaries.append(human_summary)
        
    if use_back_translation is True:
        return all_posts, all_comments, all_answers, all_human_summaries, all_posts_translated, all_comments_translated
    else:
        return all_posts, all_comments, all_answers, all_human_summaries


def batchify_data(all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, batch_size,
                  use_back_translation=False, all_posts_translated=None, all_comments_translated=None):
    comments_batches = []
    posts_batches = []
    answer_batches = []
    human_summary_batches = []
    sentences_str_batches = []

    posts_translated_batches = []
    comments_translated_batches = []

    print('Batchifying Data...')
    for i in tqdm_notebook(range(0, len(all_posts), batch_size)):
        answer_batch = []
        comments_batch = []
        post_batch = []
        human_summary_batch = []
        sentences_str_batch = []

        comments_translated_batch = []
        posts_translated_batch = []

        for j in range(i, i + batch_size):
            if j < len(all_posts):
                post = all_posts[j]#[x.replace('\n','').replace('\r','').strip() for x in data[j].initial_post]
                comments = all_comments[j]#[x.replace('\n','').replace('\r','').strip() for x in data[j].reply_sentences]
#                 selected_sentences = [x.replace('\n','').replace('\r','').strip() for x in data[j].selected_sentences]
                answers = all_answers[j]#[1 if x in selected_sentences else 0 for x in comments]
                human_summary = all_human_summaries[j]
#                 post = [hF.tokenize_text(x) for x in post]
#                 comments = [hF.tokenize_text(x) for x in comments]

                answer_batch.append(answers)
                comments_batch.append(comments)
                post_batch.append(post)
                human_summary_batch.append(human_summary)
                sentences_str_batch.append(all_sentence_str[j])

                if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
                    comments_translated_batch.append(all_comments_translated[j])
                    posts_translated_batch.append(all_posts_translated[j])

        comments_batches.append(comments_batch)
        posts_batches.append(post_batch)
        answer_batches.append(answer_batch)
        human_summary_batches.append(human_summary_batch)
        sentences_str_batches.append(sentences_str_batch)

        if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
            comments_translated_batches.append(comments_translated_batch)
            posts_translated_batches.append(posts_translated_batch)

    if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
        return posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches
    else:
        return posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches


def encode_data(data, word2id_dictionary):
    for index, doc in tqdm_notebook(enumerate(data)):
        data[index] = hF.encode_document(doc, word2id_dictionary)
    return data


def pad_data(data_batches):
    print('padding Data...')
    max_sentences = []
    max_length = []
    no_padding_sentences = []
    no_padding_lengths = []
    for index, batch in tqdm_notebook(enumerate(data_batches)):
        num_sentences = [len(x) for x in batch]
        sentence_lengthes = [[len(x) for x in y] for y in batch]
        max_num_sentences = max(num_sentences)
        max_sentences_length = max([max(x) for x in sentence_lengthes])

        batch, no_padding_num_sentences = hF.pad_batch_with_sentences(batch, max_num_sentences)
        batch, no_padding_sentence_lengths = hF.pad_batch_sequences(batch, max_sentences_length)

        max_sentences.append(max_num_sentences)
        max_length.append(max_sentences_length)
        no_padding_sentences.append(no_padding_num_sentences)
        no_padding_lengths.append(no_padding_sentence_lengths)
        data_batches[index] = batch
    ##########################################
    return data_batches, max_sentences, max_length, no_padding_sentences, no_padding_lengths


def pad_data_batch(data_batch):
    num_sentences = [len(x) for x in data_batch]
    sentence_lengthes = [[len(x) for x in y] for y in data_batch]
    max_num_sentences = max(num_sentences)
    max_sentences_length = max([max(x) for x in sentence_lengthes])

    data_batch, no_padding_num_sentences = hF.pad_batch_with_sentences(data_batch, max_num_sentences)
    data_batch, no_padding_sentence_lengths = hF.pad_batch_sequences(data_batch, max_sentences_length)

    ##########################################
    return data_batch, max_num_sentences, max_sentences_length, no_padding_num_sentences, no_padding_sentence_lengths


def encode_and_pad_data(data_batches, word2id_dictionary):
    #################### Prepare Training data################
    print('Encoding Data...')
    max_sentences = []
    max_length = []
    no_padding_sentences = []
    no_padding_lengths = []
    for index, batch in tqdm_notebook(enumerate(data_batches)):
        batch = hF.encode_batch(batch, word2id_dictionary)

        num_sentences = [len(x) for x in batch]
        sentence_lengthes = [[len(x) for x in y] for y in batch]
        max_num_sentences = max(num_sentences)
        max_sentences_length = max([max(x) for x in sentence_lengthes])

        batch, no_padding_num_sentences = hF.pad_batch_with_sentences(batch, max_num_sentences)
        batch, no_padding_sentence_lengths = hF.pad_batch_sequences(batch, max_sentences_length)

        max_sentences.append(max_num_sentences)
        max_length.append(max_sentences_length)
        no_padding_sentences.append(no_padding_num_sentences)
        no_padding_lengths.append(no_padding_sentence_lengths)
        data_batches[index] = batch
    ##########################################
    return data_batches, max_sentences, max_length, no_padding_sentences, no_padding_lengths


def encode_data_BERT(data, Bert_model_Path, device, bert_layers, batch_size):
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    if not os.path.exists(Bert_model_Path):
        print('Bet Model not found.. make sure path is correct')
        return
    tokenizer = BertTokenizer.from_pretrained(Bert_model_Path)#'../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
    model = BertModel.from_pretrained(Bert_model_Path)#'../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
    model.eval()
    model.to(device)
    #################### Prepare Training data################
    print('Encoding Data using BERT...')
    max_sentences = []
    no_padding_sentences = []
    j = 0
    for j in tqdm_notebook(range(0, len(data), batch_size)):
        if j + batch_size < len(data):
            batch = data[j: j + batch_size]
        else:
            batch = data[j:]
        batch = hF.encode_batch_BERT(batch, model, tokenizer, device, bert_layers)

        for i, doc in enumerate(batch):
            data[j+i] = batch[i]

    ##########################################
    return data


def pad_data_BERT(data_batches, bert_layers, bert_dims):
    print('Padding Data using BERT...')
    max_sentences = []
    no_padding_sentences = []
    for index, batch in tqdm_notebook(enumerate(data_batches)):
        num_sentences = [len(x) for x in batch]
        max_num_sentences = max(num_sentences)

        batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)

        max_sentences.append(max_num_sentences)
        no_padding_sentences.append(no_padding_num_sentences)
        data_batches[index] = batch
    ##########################################
    return data_batches, max_sentences, None, no_padding_sentences, None


def pad_batch_BERT(batch, bert_layers, bert_dims):
    num_sentences = [len(x) for x in batch]
    max_num_sentences = max(num_sentences)
    batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)
    ##########################################
    return batch, max_num_sentences, None, no_padding_num_sentences, None


def encode_and_pad_data_BERT(data_batches, Bert_model_Path, device, bert_layers, bert_dims):
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(Bert_model_Path)#'../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
    model = BertModel.from_pretrained(Bert_model_Path)#'../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
    model.eval()
    model.to(device)
    #################### Prepare Training data################
    print('Encoding Data using BERT...')
    max_sentences = []
    no_padding_sentences = []
    for index, batch in tqdm_notebook(enumerate(data_batches)):
        batch = hF.encode_batch_BERT(batch, model, tokenizer, device, bert_layers)
        # data_batches[index] = batch
        num_sentences = [len(x) for x in batch]
        max_num_sentences = max(num_sentences)

        batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)

        max_sentences.append(max_num_sentences)
        no_padding_sentences.append(no_padding_num_sentences)
        data_batches[index] = batch
    ##########################################
    return data_batches, max_sentences, None, no_padding_sentences, None

if __name__ == '__main__':
    all_data, word2id_dictionary, id2word_dictionary = load_data('./forum_data/data_V2/Parsed_Data.xml')
                                                                 # , use_back_translation=True,
                                                                 # back_translation_file='E:/Work/Summarization_samples/SummRunner_V2/checkpoint/forum_to_translate_en.txt')
    del all_data
    del word2id_dictionary
    del id2word_dictionary

    all_data, word2id_dictionary, id2word_dictionary = load_cnn_dm_data('./cnn_data/finished_files/')
    del all_data
    del word2id_dictionary
    del id2word_dictionary
