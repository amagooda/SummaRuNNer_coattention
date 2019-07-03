import constants
import torch
from nltk import word_tokenize
from nltk import sent_tokenize


def sent_tokenize_text(text):
    sents = sent_tokenize(text)
    sents = [sent.strip() for sent in sents if sent.strip() != '']
    return sents


def tokenize_text(text):
    words = word_tokenize(text)
    words = [word.strip() for word in words if word.strip() != '']
    return words


def encode_batch(batch_documents, dictionary):
    for index, doc in enumerate(batch_documents):
        batch_documents[index] = encode_document(doc, dictionary)
    return batch_documents


def encode_document(input_document, dictionary):
    for index, sequence in enumerate(input_document):
        input_document[index] = encode_sequence(sequence, dictionary)
    return input_document


def encode_sequence(sequence, dictionary):
    if sequence[0] != '<SOS>':
        sequence = ['<SOS>'] + sequence
    if sequence[-1] != '<EOS>':
        sequence = sequence + ['<EOS>']
    sequence_in_indcies = [dictionary[word] if word in dictionary else constants.oov_index for word in sequence]
    return sequence_in_indcies


def pad_batch_sequences(batch, max_sentence_len):
    batch_sequence_lengths = []
    for index, doc in enumerate(batch):
        batch[index], doc_sequence_length = pad_document_sequences(doc, max_sentence_len)
        batch_sequence_lengths.append(doc_sequence_length)
    return batch, batch_sequence_lengths


def pad_document_sequences(docs, max_sentence_len):
    doc_sequence_lengths = []
    for index, sequence in enumerate(docs):
        docs[index], sequence_length = pad_sequence(sequence, max_sentence_len)
        doc_sequence_lengths.append(sequence_length)
    return docs, doc_sequence_lengths


def pad_sequence(input_sequence, max_sentence_len):
    sequence_len = len(input_sequence)
    pad_length = max_sentence_len - sequence_len
    input_sequence = input_sequence + [constants.pad_index for i in range(pad_length)]
    return input_sequence, sequence_len


def pad_batch_with_sentences(batch, max_num_sentences):
    batch_sentences_count = []
    for index, doc in enumerate(batch):
        batch[index], doc_num_sentences = pad_doc_with_sentences(doc, max_num_sentences)
        batch_sentences_count.append(doc_num_sentences)
    return batch, batch_sentences_count


def pad_doc_with_sentences(input_document, max_num_sentences):
    num_sentences = len(input_document)
    pad_length = max_num_sentences - num_sentences
    input_document = input_document + [[constants.pad_index] for i in range(pad_length)]
    return input_document, num_sentences


def convert_to_tensor(sequences, device, data_type='long'):
    if device is not None and device.type == 'cuda':
        if data_type == 'long':
            sequence_tensor = torch.tensor(sequences, dtype=torch.long).cuda()
        else:
            sequence_tensor = torch.tensor(sequences, dtype=torch.float).cuda()
    else:
        if data_type == 'long':
            sequence_tensor = torch.tensor(sequences, dtype=torch.long)
        else:
            sequence_tensor = torch.tensor(sequences, dtype=torch.float)
    return sequence_tensor


def encode_batch_BERT(batch, bert_model, bert_tokenizer, device, bert_layers):
    import numpy as np

    encoded_batch = []
    for document in batch:
        document_segments = []
        document_indcies = []
        document_masks = []
        max_sentence_length = 0
        for sentence in document:
            tokenized_text = bert_tokenizer.tokenize(' '.join(['[CLS]'] + sentence + ['[SEP]']))
            # Convert token to vocabulary indices
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [0 for word in tokenized_text]
            mask = [1 for word in tokenized_text]

            if len(segments_ids) > max_sentence_length:
                max_sentence_length = len(segments_ids)
            document_segments.append(segments_ids)
            document_indcies.append(indexed_tokens)
            document_masks.append(mask)

        ############### PAD
        for index, _ in enumerate(document_indcies):
            while len(document_indcies[index]) < max_sentence_length:
                document_indcies[index].append(0)
                document_segments[index].append(0)
                document_masks[index].append(0)

        ###################
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(document_indcies, device=device)
        segments_tensors = torch.tensor(document_segments, device=device)
        masks_tensors = torch.tensor(document_masks,device=device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = bert_model(tokens_tensor, segments_tensors, attention_mask=masks_tensors)

        encoded_layers = [x.tolist() for x in encoded_layers]

        doc_embeddings = [[] for i in bert_layers]
        for (j, layer_index) in enumerate(bert_layers):
            layer = encoded_layers[layer_index]

            for sent_index, sentence in enumerate(layer):
                sent_embedding = []
                for token_index, token in enumerate(sentence):
                    if document_masks[sent_index][token_index] == 1:
                        sent_embedding.append(token)
                doc_embeddings[j].append(np.average(sent_embedding, 0).tolist())

        concat_doc_embedding = []
        for sent_index, sent in enumerate(doc_embeddings[0]):
            concat_sent_embedding = []
            for i in bert_layers:
                concat_sent_embedding += doc_embeddings[i][sent_index]
            concat_doc_embedding.append(concat_sent_embedding)

        encoded_batch.append(concat_doc_embedding)
    return encoded_batch

def pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims):
    batch_sentences_count = []
    for index, doc in enumerate(batch):
        batch[index], doc_num_sentences = pad_doc_with_sentences_BERT(doc, max_num_sentences, bert_layers, bert_dims)
        batch_sentences_count.append(doc_num_sentences)
    return batch, batch_sentences_count


def pad_doc_with_sentences_BERT(input_document, max_num_sentences, bert_layers, bert_dims):
    num_sentences = len(input_document)
    pad_length = max_num_sentences - num_sentences
    input_document = input_document + [[constants.pad_index for j in range(len(bert_layers) * bert_dims)] for i in range(pad_length)]
    return input_document, num_sentences