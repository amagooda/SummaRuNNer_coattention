from tqdm  import tqdm as tqdm_notebook
import torch
import codecs
import HelpingFunctions as HelpingFunctions
import os
from torch.nn.utils import clip_grad_norm_
import copy



def train_batch(model, device, post_batch, comment_batch, answer_batch,
                max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                optimizer, criterion, use_bert):
    # model.train()
    # epoch_loss = 0
    # pbar = tqdm_notebook(enumerate(comment_batches))
    # for index, batch in pbar:
    #     pbar.set_description("Training {}/{}, loss={}".format(index, len(comment_batches), round(epoch_loss)))
#         tensor_batch = batch.to(device)#HelpingFunctions.convert_to_tensor(batch, device)
#         tensor_post_batch = post_batches[index].to(device)#HelpingFunctions.convert_to_tensor(post_batches[index], device)

    if use_bert is True:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device, 'float')
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device, 'float')
    else:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device)
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device)

    if max_length is not None:
        batch_max_length = max_length
    else:
        batch_max_length = None

    if no_padding_lengths is not None:
        batch_no_padding_lengths = no_padding_lengths
    else:
        batch_no_padding_lengths = None

    if posts_max_length is not None:
        batch_posts_max_length = posts_max_length
    else:
        batch_posts_max_length = None

    if posts_no_padding_lengths is not None:
        batch_posts_no_padding_lengths = posts_no_padding_lengths
    else:
        batch_posts_no_padding_lengths = None


    sentence_probabilities = model(tensor_batch, max_sentences, batch_max_length, no_padding_sentences, batch_no_padding_lengths,
                                   tensor_post_batch, posts_max_sentences, batch_posts_max_length, posts_no_padding_sentences, batch_posts_no_padding_lengths)

    # for i in range(len(sentence_probabilities)):
    #     for j in range(len(sentence_probabilities[i,:])):
    #         if j >= no_padding_sentences[i]:
    #             sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

    # targets = copy.deepcopy(answer_batch)
    # for i, elem in enumerate(targets):
    #     while len(targets[i]) < max_sentences:
    #         targets[i].append(0)

    targets = copy.deepcopy(answer_batch)
    for i, elem in enumerate(targets):
        while len(targets[i]) < max_sentences:
            targets[i].append(0)
    for i in range(len(sentence_probabilities)):
        for j in range(len(sentence_probabilities[i,:])):
            if j >= no_padding_sentences[i]:
                targets[i][j] = sentence_probabilities[i, j].item()

    # loss_cri = torch.nn.BCELoss(reduction='sum')
    # loss = criterion(sentence_probabilities, torch.FloatTensor(targets).to(device))
    loss = criterion(sentence_probabilities, torch.FloatTensor(targets).to(device))
    loss = loss/sum(no_padding_sentences)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 2)
    optimizer.step()
    return loss.item()
    # epoch_loss += loss.item()
# epoch_loss = epoch_loss / len(comment_batches)
#     # print('Epch {}:\t{}'.format(epoch, epoch_loss))
#     return epoch_loss



def val_batch(model, device, post_batch, comment_batch, answer_batch,
                max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                criterion, use_bert):
    if use_bert is True:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device, 'float')
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device, 'float')
    else:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device)
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device)

    if max_length is not None:
        batch_max_length = max_length
    else:
        batch_max_length = None

    if no_padding_lengths is not None:
        batch_no_padding_lengths = no_padding_lengths
    else:
        batch_no_padding_lengths = None

    if posts_max_length is not None:
        batch_posts_max_length = posts_max_length
    else:
        batch_posts_max_length = None

    if posts_no_padding_lengths is not None:
        batch_posts_no_padding_lengths = posts_no_padding_lengths
    else:
        batch_posts_no_padding_lengths = None

    sentence_probabilities = model(tensor_batch, max_sentences,
                                   batch_max_length, no_padding_sentences, batch_no_padding_lengths,
                                   tensor_post_batch, posts_max_sentences, batch_posts_max_length,
                                   posts_no_padding_sentences, batch_posts_no_padding_lengths)

    # for i in range(len(sentence_probabilities)):
    #     for j in range(len(sentence_probabilities[i,:])):
    #         if j >= no_padding_sentences[i]:
    #             sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

    # targets = answer_batch
    # for i, elem in enumerate(targets):
    #     while len(targets[i]) < max_sentences:
    #         targets[i].append(0)

    targets = copy.deepcopy(answer_batch)
    for i, elem in enumerate(targets):
        while len(targets[i]) < max_sentences:
            targets[i].append(0)

    for i in range(len(sentence_probabilities)):
        for j in range(len(sentence_probabilities[i,:])):
            if j >= no_padding_sentences[i]:
                targets[i][j] = sentence_probabilities[i, j].item()

    loss = criterion(sentence_probabilities, torch.FloatTensor(targets).to(device))
    loss = loss / sum(no_padding_sentences)
    return loss.item()


def test_batch(model, device, post_batch, comment_batch, answer_batch, human_summary_batch, sentences_str_batch,
               max_sentences, max_length, no_padding_sentences, no_padding_lengths,
               posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths, use_bert):
    if use_bert is True:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device, 'float')
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device, 'float')
    else:
        tensor_batch = HelpingFunctions.convert_to_tensor(comment_batch, device)
        tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batch, device)

    if max_length is not None:
        batch_max_length = max_length
    else:
        batch_max_length = None

    if no_padding_lengths is not None:
        batch_no_padding_lengths = no_padding_lengths
    else:
        batch_no_padding_lengths = None

    if posts_max_length is not None:
        batch_posts_max_length = posts_max_length
    else:
        batch_posts_max_length = None

    if posts_no_padding_lengths is not None:
        batch_posts_no_padding_lengths = posts_no_padding_lengths
    else:
        batch_posts_no_padding_lengths = None

    sentence_probabilities = model(tensor_batch, max_sentences,
                                   batch_max_length, no_padding_sentences, batch_no_padding_lengths,
                                   tensor_post_batch, posts_max_sentences, batch_posts_max_length,
                                   posts_no_padding_sentences, batch_posts_no_padding_lengths)

    for i in range(len(sentence_probabilities)):
        for j in range(len(sentence_probabilities[i, :])):
            if j >= no_padding_sentences[i]:
                sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

    sentence_probabilities = sentence_probabilities.tolist()
    targets = answer_batch
    human_summaries = human_summary_batch

    target_words = []
    predicted_words = []
    # human_summaries = []
    for target, prediction, indcies in zip(targets, sentence_probabilities, sentences_str_batch):
        ## for each document in the batch
        pre_sentences = []
        tar_sentences = []
        for i, val in enumerate(target):
            if target[i] == 1:
                tar_sentences.append(indcies[i])
            if prediction[i] > 0.5:
                pre_sentences.append(indcies[i])

        target_words.append(tar_sentences)
        predicted_words.append(pre_sentences)

    target_sentences = []
    predicted_sentences = []
    for doc in target_words:
        doc_text = ''
        for sentence in doc:
            sentence_text = ' '.join(sentence).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            doc_text += sentence_text
        target_sentences.append(doc_text)

    for index, human_summary in enumerate(human_summaries):
        human_summaries[index] = human_summary.replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()

    for doc in predicted_words:
        doc_text = ''
        for sentence in doc:
            sentence_text = ' '.join(sentence).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            doc_text += sentence_text
        predicted_sentences.append(doc_text)

    return predicted_sentences, target_sentences, human_summaries


def train_epoch(model, device, post_batches, comment_batches, answer_batches,
                max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                optimizer, criterion, use_bert):
    model.train()
    epoch_loss = 0
    pbar = tqdm_notebook(enumerate(comment_batches))
    for index, batch in pbar:
        pbar.set_description("Training {}/{}, loss={}".format(index, len(comment_batches), round(epoch_loss)))
#         tensor_batch = batch.to(device)#HelpingFunctions.convert_to_tensor(batch, device)
#         tensor_post_batch = post_batches[index].to(device)#HelpingFunctions.convert_to_tensor(post_batches[index], device)

        if use_bert is True:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device, 'float')
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device, 'float')
        else:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device)
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device)

        if max_length is not None:
            batch_max_length = max_length[index]
        else:
            batch_max_length = None

        if no_padding_lengths is not None:
            batch_no_padding_lengths = no_padding_lengths[index]
        else:
            batch_no_padding_lengths = None

        if posts_max_length is not None:
            batch_posts_max_length = posts_max_length[index]
        else:
            batch_posts_max_length = None

        if posts_no_padding_lengths is not None:
            batch_posts_no_padding_lengths = posts_no_padding_lengths[index]
        else:
            batch_posts_no_padding_lengths = None

        sentence_probabilities = model(tensor_batch, max_sentences[index], batch_max_length, no_padding_sentences[index], batch_no_padding_lengths,
                                       tensor_post_batch, posts_max_sentences[index], batch_posts_max_length, posts_no_padding_sentences[index], batch_posts_no_padding_lengths)

        for i in range(len(sentence_probabilities)):
            for j in range(len(sentence_probabilities[i,:])):
                if j >= no_padding_sentences[index][i]:
                    sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

        targets = answer_batches[index]
        for i, elem in enumerate(targets):
            while len(targets[i]) < max_sentences[index]:
                targets[i].append(0)

        loss = criterion(sentence_probabilities, torch.FloatTensor(targets).to(device))
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(comment_batches)
    # print('Epch {}:\t{}'.format(epoch, epoch_loss))
    return epoch_loss


def validate_epoch(model, device, post_batches, comment_batches, answer_batches, 
                   max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                   posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
                   criterion, use_bert):
    model.eval()
    val_loss = 0
    pbar = tqdm_notebook(enumerate(comment_batches))
    for index, batch in pbar:
        pbar.set_description("Validating {}/{}, loss={}".format(index, len(comment_batches), round(val_loss)))
#         tensor_batch = batch.to(device)#HelpingFunctions.convert_to_tensor(batch, device)
#         tensor_post_batch = post_batches[index].to(device)#HelpingFunctions.convert_to_tensor(post_batches[index], device)
        
        if use_bert is True:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device, 'float')
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device, 'float')
        else:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device)
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device)
        
        if max_length is not None:
            batch_max_length = max_length[index]
        else:
            batch_max_length = None

        if no_padding_lengths is not None:
            batch_no_padding_lengths = no_padding_lengths[index]
        else:
            batch_no_padding_lengths = None

        if posts_max_length is not None:
            batch_posts_max_length = posts_max_length[index]
        else:
            batch_posts_max_length = None

        if posts_no_padding_lengths is not None:
            batch_posts_no_padding_lengths = posts_no_padding_lengths[index]
        else:
            batch_posts_no_padding_lengths = None

        sentence_probabilities = model(tensor_batch, max_sentences[index], batch_max_length, no_padding_sentences[index], batch_no_padding_lengths,
                                       tensor_post_batch, posts_max_sentences[index], batch_posts_max_length, posts_no_padding_sentences[index], batch_posts_no_padding_lengths)

        for i in range(len(sentence_probabilities)):
            for j in range(len(sentence_probabilities[i,:])):
                if j >= no_padding_sentences[index][i]:
                    sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

        targets = answer_batches[index]
        for i, elem in enumerate(targets):
            while len(targets[i]) < max_sentences[index]:
                targets[i].append(0)
        loss = criterion(sentence_probabilities, torch.FloatTensor(targets).to(device))
        val_loss += loss.item()
    val_loss = val_loss / len(answer_batches)
    # print('Validation Loss:\t{}'.format(val_loss))
    return val_loss


def test_epoch(model, device, post_batches, comment_batches, answer_batches, human_summary_batches, sentences_str_batches,
               max_sentences, max_length, no_padding_sentences, no_padding_lengths,
               posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
               id2word_dic, output_dir, use_bert):
    model.eval()

    target_words = []
    predicted_words = []
    human_summaries = []
    
    pbar = tqdm_notebook(enumerate(comment_batches))
    for index, batch in pbar:
        pbar.set_description("Testing {}/{}".format(index, len(comment_batches)))
#         tensor_batch = batch.to(device)#HelpingFunctions.convert_to_tensor(batch, device)
#         tensor_post_batch = post_batches[index].to(device)#HelpingFunctions.convert_to_tensor(post_batches[index], device)
        
        if use_bert is True:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device, 'float')
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device, 'float')
        else:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device)
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device)
        
        if max_length is not None:
            batch_max_length = max_length[index]
        else:
            batch_max_length = None

        if no_padding_lengths is not None:
            batch_no_padding_lengths = no_padding_lengths[index]
        else:
            batch_no_padding_lengths = None

        if posts_max_length is not None:
            batch_posts_max_length = posts_max_length[index]
        else:
            batch_posts_max_length = None

        if posts_no_padding_lengths is not None:
            batch_posts_no_padding_lengths = posts_no_padding_lengths[index]
        else:
            batch_posts_no_padding_lengths = None

        sentence_probabilities = model(tensor_batch, max_sentences[index], batch_max_length, no_padding_sentences[index], batch_no_padding_lengths,
                                       tensor_post_batch, posts_max_sentences[index], batch_posts_max_length, posts_no_padding_sentences[index], batch_posts_no_padding_lengths)

        # for i in range(len(sentence_probabilities)):
        #     for j in range(len(sentence_probabilities[i,:])):
        #         if j >= no_padding_sentences[index][i]:
        #             sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

        sentence_probabilities = sentence_probabilities.tolist()
        targets = answer_batches[index]
        human_summaries += human_summary_batches[index]

        for target, prediction, indcies in zip(targets, sentence_probabilities, sentences_str_batches[index]):
            pre_sentences = []
            tar_sentences = []
            for i, val in enumerate(target):
                if target[i] == 1:
                    tar_sentences.append(indcies[i])
                if prediction[i] > 0.5:
                    pre_sentences.append(indcies[i])

            target_words.append(tar_sentences)
            predicted_words.append(pre_sentences)

    index = 1
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 
    if not os.path.exists(output_dir + '/ref/'):
        os.mkdir(output_dir + '/ref/')
    if not os.path.exists(output_dir + '/ref_abs/'):
        os.mkdir(output_dir + '/ref_abs/')
    if not os.path.exists(output_dir + '/dec/'):
        os.mkdir(output_dir + '/dec/')

    for target_elem, prediction_elem, human_summary in zip(target_words, predicted_words, human_summaries):
        gold_abs_output = codecs.open(output_dir + '/ref_abs/{}.ref'.format(index), 'w', encoding='utf8')
        gold_output = codecs.open(output_dir + '/ref/{}.ref'.format(index), 'w', encoding='utf8')
        predicted_output = codecs.open(output_dir + '/dec/{}.dec'.format(index), 'w', encoding='utf8')

        for sentence in target_elem:
            #sentence_text = ' '.join([id2word_dic[word] for word in sentence]).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            sentence_text = ' '.join(sentence).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            gold_output.write(sentence_text + '\n')
        gold_output.close()
        

        gold_abs_output.write(human_summary.replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip())
        gold_abs_output.close()

        for sentence in prediction_elem:
            #sentence_text = ' '.join([id2word_dic[word] for word in sentence]).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            sentence_text = ' '.join(sentence).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            predicted_output.write(sentence_text + '\n')
        predicted_output.close()

        index += 1


def summarize(model, device, post_batches, test_comment_batches, test_human_summary_batches, sentences_str_batches,
              max_sentences, max_length, no_padding_sentences, no_padding_lengths,
              posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths,
              id2word_dic, output_dir, use_bert):
    model.eval()
    predicted_words = []

    for index, batch in tqdm_notebook(enumerate(test_comment_batches)):
#         tensor_batch = batch.to(device)#HelpingFunctions.convert_to_tensor(batch, device)
#         tensor_post_batch = post_batches[index].to(device)#HelpingFunctions.convert_to_tensor(post_batches[index], device)
        
        if use_bert is True:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device, 'float')
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device, 'float')
        else:
            tensor_batch = HelpingFunctions.convert_to_tensor(batch, device)
            tensor_post_batch = HelpingFunctions.convert_to_tensor(post_batches[index], device)


        if max_length is not None:
            batch_max_length = max_length[index]
        else:
            batch_max_length = None

        if no_padding_lengths is not None:
            batch_no_padding_lengths = no_padding_lengths[index]
        else:
            batch_no_padding_lengths = None

        if posts_max_length is not None:
            batch_posts_max_length = posts_max_length[index]
        else:
            batch_posts_max_length = None

        if posts_no_padding_lengths is not None:
            batch_posts_no_padding_lengths = posts_no_padding_lengths[index]
        else:
            batch_posts_no_padding_lengths = None

        sentence_probabilities = model(tensor_batch, max_sentences[index], batch_max_length, no_padding_sentences[index], batch_no_padding_lengths,
                                       tensor_post_batch, posts_max_sentences[index], batch_posts_max_length, posts_no_padding_sentences[index], batch_posts_no_padding_lengths)

        for i in range(len(sentence_probabilities)):
            for j in range(len(sentence_probabilities[i,:])):
                if j >= no_padding_sentences[index][i]:
                    sentence_probabilities[i, j] = 0 * sentence_probabilities[i, j]

        sentence_probabilities = sentence_probabilities.tolist()


        for prediction, indcies in zip(sentence_probabilities, sentences_str_batches[index]):
            pre_tokens = []
            for i, val in enumerate(prediction):
                if prediction[i] > 0.5:
                    pre_tokens.append(indcies[i])
            predicted_words.append(pre_tokens)

    index = 1
    if not os.path.exists(output_dir + '/dec/'):
        os.mkdir(output_dir + '/dec/')

    for prediction_elem in predicted_words:
        predicted_output = codecs.open(output_dir + '/dec/{}.dec'.format(index), 'w', encoding='utf8')

        for sentence in prediction_elem:
            # sentence_text = ' '.join([id2word_dic[word] for word in sentence]).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            sentence_text = ' '.join(sentence).replace('<SOS>', '').replace('<EOS>', '').replace('<pad>', '').strip()
            predicted_output.write(sentence_text + '\n')
        predicted_output.close()

        index += 1
