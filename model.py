import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class EncoderBiRNN(nn.Module):
    def __init__(self, batch_size, vocab_size, embedding_size,
                 hidden_size, max_num_sentence=360, device=None,
                 pretrained_embeddings=None, pretrained_path=None, use_bert=False,
                 num_bert_layers=3, bert_embedding_size=768, use_coattention=True):
        super(EncoderBiRNN, self).__init__()
        self.num_documents_dim = 0
        self.num_sentences_dim = 1
        self.num_tokens_dim = 2

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sent_num = max_num_sentence
        self.positional_embedding_size = 50

        self.use_BERT = use_bert
        self.use_coattention = use_coattention

        self.layers = []

        if self.use_BERT is True:
            print('using Bert')
            self.embedding_size = num_bert_layers * bert_embedding_size

            #   2.    /sentence and doc embedding layers
            self.word_BiLSTM = None
            self.sentence_BiLSTM = nn.LSTM(input_size=self.embedding_size, hidden_size= hidden_size, bidirectional=True, batch_first=True)

            self.layers.append(self.sentence_BiLSTM)
        else:
            #   1.    embedding layer
            self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

            if pretrained_embeddings is not None:
                self.load_pretrained_embedding(pretrained_path)
            self.layers.append(self.embedding_layer)

            #   2.    /sentence and doc embedding layers
            self.word_BiLSTM = nn.LSTM(input_size=embedding_size, hidden_size= hidden_size, bidirectional=True, batch_first=True)
            self.sentence_BiLSTM = nn.LSTM(input_size=2 * hidden_size, hidden_size= hidden_size, bidirectional=True, batch_first=True)
            self.layers.append(self.word_BiLSTM)
            self.layers.append(self.sentence_BiLSTM)

        if self.use_coattention is True:
            #   3. Attention Flow Layer
            self.att_weight_c = Linear(hidden_size * 2, 1)
            self.att_weight_q = Linear(hidden_size * 2, 1)
            self.att_weight_cq = Linear(hidden_size * 2, 1)

            self.layers.append(self.att_weight_c)
            self.layers.append(self.att_weight_q)
            self.layers.append(self.att_weight_cq)

        # self.attention = Attention(0.5)
        # self.word_query = nn.Parameter(torch.randn(1, 1, 2 * hidden_size))
        # self.sentence_query = nn.Parameter(torch.randn(1, 1, 2 * hidden_size))

        #   4. Positional Embedding
        # S = 15
        self.abs_pos_embed = nn.Embedding(self.max_sent_num, self.positional_embedding_size)
        self.layers.append(self.abs_pos_embed)

        # self.rel_pos_embed = nn.Embedding(S,positional_embedding_size)

        #   5.   Output Layer
        self.content = nn.Linear(2*hidden_size, 1, bias=False)
        self.layers.append(self.content)
        self.salience = nn.Bilinear(2*hidden_size, 2*hidden_size, 1, bias=False)
        self.layers.append(self.salience)
        self.novelty = nn.Bilinear(2*hidden_size, 2*hidden_size, 1, bias=False)
        self.layers.append(self.novelty)
        if self.use_coattention is True:
            self.attention_and_query = nn.Linear(12 * hidden_size, hidden_size, bias=False)
            self.layers.append(self.attention_and_query)
            self.prob_attention = nn.Linear(hidden_size, 1, bias=False)
            self.layers.append(self.prob_attention)
        self.abs_pos = nn.Linear(self.positional_embedding_size, 1, bias=False)
        self.layers.append(self.abs_pos)
        # self.rel_pos = nn.Linear(positional_embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))


        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def load_pretrained_embedding(self, pretrained_path):
        self.embedding_layer = nn.Embedding.from_pretrained(pretrained_path, freeze=True)

    def reinit_embeddings(self, vocab_size, embedding_size, max_sent_num):
        if embedding_size == self.embedding_size:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.abs_pos_embed = nn.Embedding(max_sent_num, self.positional_embedding_size)
        else:
            print('can\'t reinit embeddings with different embedding dimesions, you need to use embedding with size {} or reinitialize the whole model'.format(self.embedding_size))

    def avg_pool1d(self, input, seq_lens):
        out = []
        for index, t in enumerate(input):
            t = t[:seq_lens[index], :]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, comments, max_num_sentences, max_len, num_sentences, sequnce_lens, post, post_max_num_sentences, post_max_len, post_num_sentences, post_sequnce_lens):
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def sentence_embedding_layer(_word_embeddings, _seq_lengths):
            _sents_emb = None
            for i in range(len(_word_embeddings[0, :, 0, 0])):
                c_w_embeddings, _ = self.word_BiLSTM(_word_embeddings[:, i, :, :])
                sent_emb = self.avg_pool1d(c_w_embeddings, [x[i] for x in _seq_lengths])
                sent_emb = sent_emb.unsqueeze(1)
                if _sents_emb is None:
                    _sents_emb = sent_emb
                else:
                    _sents_emb = torch.cat((_sents_emb, sent_emb), 1)
            return _sents_emb

        def doc_embedding_layer(sent_embeddings, _num_sentences):
            c_s_embeddings, _ = self.sentence_BiLSTM(sent_embeddings)
            _doc_embedding = self.avg_pool1d(c_s_embeddings, _num_sentences)
            _doc_embedding = torch.tanh(_doc_embedding)  # .unsqueeze(1)
            return _doc_embedding, c_s_embeddings

        doc_count = comments.size(0)
        sent_count = comments.size(1)
        seq_len = comments.size(2)

        #################  Process comments  ###############
        if self.use_BERT is True:
            doc_embedding, bi_sent_embeddings = doc_embedding_layer(comments, num_sentences)
            #################  Process post  #############################################
            post_doc_embedding, post_bi_sent_embeddings = doc_embedding_layer(post, post_num_sentences)
            #####################################################################################
        else:
            word_embeddings = self.embedding_layer(comments)
            ''' number of sentences in document =  len(word_embeddings[0,:,0,0]) '''
            ''' loop over each sentence position, get the sentence representation for each sentence by pooling the word representations'''
            ''' using the sentence representations get the document representation'''
            sentence_embeddings = sentence_embedding_layer(word_embeddings, sequnce_lens)
            doc_embedding, bi_sent_embeddings = doc_embedding_layer(sentence_embeddings, num_sentences)


            #################  Process post  #############################################
            post_embeddings = self.embedding_layer(post)

            post_sentence_embeddings = sentence_embedding_layer(post_embeddings, post_sequnce_lens)
            post_doc_embedding, post_bi_sent_embeddings = doc_embedding_layer(post_sentence_embeddings, post_num_sentences)
            #####################################################################################
        if self.use_coattention is True:
            attention_vec = att_flow_layer(bi_sent_embeddings, post_bi_sent_embeddings)

        s = Variable(torch.zeros(doc_count, 2 * self.hidden_size))
        if self.device.type == 'cuda':
            s = s.cuda()
        probs = []
        for i in range(len(bi_sent_embeddings[0,:,0])):
            ######## Positional feature #########
            abs_index = Variable(torch.LongTensor([[i + 1] if i <= num_sentences[j] else [0] for j in range(doc_count)]))
            if self.device.type == 'cuda':
                abs_index = abs_index.cuda()
            abs_features = self.abs_pos_embed(abs_index).squeeze(1)
            ################
            if self.use_coattention is True:
                attention_and_initial_comment = self.attention_and_query(torch.cat((attention_vec[:,i,:], doc_embedding, post_doc_embedding),1))#doc_embedding, post_doc_embedding)
                attn_prob = self.prob_attention(attention_and_initial_comment)

            content = self.content(bi_sent_embeddings[:,i,:])
            salience = self.salience(bi_sent_embeddings[:,i,:], doc_embedding)
            novelty = -1 * self.novelty(bi_sent_embeddings[:,i,:], torch.tanh(s))
            abs_p = self.abs_pos(abs_features)
            # rel_p = self.rel_pos(rel_features)
            if self.use_coattention is True:
                prob = torch.sigmoid(content + salience + novelty + abs_p + attn_prob + self.bias)
            else:
                prob = torch.sigmoid(content + salience + novelty + abs_p + self.bias)
            s = s + prob * bi_sent_embeddings[:,i,:]
            probs.append(prob)
        return torch.cat(probs,1)










# if __name__ == '__main__':
#     torch.manual_seed(1)
#     attention = Attention()
#     context = Variable(torch.randn(10, 20, 4))
#     query = Variable(torch.randn(10, 1, 4))
#     query, attn = attention(query, context)
#     print(query)







        # doc_count = comments.size(0)
        # sent_count = comments.size(1)
        # seq_len = comments.size(2)
        #
        # #################  Process comments  ###############
        # word_embeddings = self.embedding_layer(comments)
        # sentence_embeddings = None
        #
        # ''' number of sentences in document =  len(word_embeddings[0,:,0,0]) '''
        # ''' loop over each sentence position, get the sentence representation for each sentence by pooling the word representations'''
        # ''' using the sentence representations get the document representation'''
        # for i in range(len(word_embeddings[0,:,0,0])):
        #     bi_word_embeddings, final_hidden_state = self.word_BiLSTM(word_embeddings[:,i,:,:])
        #
        #     # ####### attention
        #     # word_mask = torch.ones_like(comments[:, i, :]) - torch.sign(comments[:, i, :])
        #     # word_mask = word_mask.data.type(torch.ByteTensor).view(doc_count, seq_len, 1)
        #     # if self.device.type == 'cuda':
        #     #     word_mask = word_mask.cuda()
        #     #
        #     # query = self.word_query.expand(doc_count, seq_len, -1).contiguous()
        #     # self.attention.set_mask(word_mask)
        #     # word_out = self.attention(query, bi_word_embeddings)[0].squeeze(1)  # (N,2*H)
        #     # #########################
        #     # #use_attention
        #     # booled_words = self.avg_pool1d(word_out, [x[i] for x in sequnce_lens])
        #     booled_words = self.avg_pool1d(bi_word_embeddings, [x[i] for x in sequnce_lens])
        #     booled_words = booled_words.unsqueeze(1)
        #     if sentence_embeddings is None:
        #         sentence_embeddings = booled_words
        #     else:
        #         sentence_embeddings = torch.cat((sentence_embeddings, booled_words), 1)
        #
        # bi_sentence_embeddings, doc_embedding = self.sentence_BiLSTM(sentence_embeddings)
        #
        # ####### sentence attention
        # # sentence_mask = torch.ones_like(comments[:, :, 0]) - torch.sign(comments[:, :, 0])
        # # sentence_mask = sentence_mask.data.type(torch.ByteTensor).view(doc_count, sent_count, 1)
        # # if self.device.type == 'cuda':
        # #     sentence_mask = sentence_mask.cuda()
        # #
        # # query = self.sentence_query.expand(doc_count, sent_count, -1).contiguous()
        # # self.attention.set_mask(sentence_mask)
        # # sentence_out = self.attention(query, bi_sentence_embeddings)[0].squeeze(1)  # (N,2*H)
        # #########################
        # # use_attention
        # # booled_sentences = self.avg_pool1d(sentence_out, num_sentences)
        # booled_sentences = self.avg_pool1d(bi_sentence_embeddings, num_sentences)
        # booled_sentences = torch.tanh(booled_sentences)#.unsqueeze(1)
        #
        #
        #
        # #################  Process post  #############################################
        # post_embeddings = self.embedding_layer(post)
        # post_sentence_embeddings = None
        #
        # for i in range(len(post_embeddings[0,:,0,0])):
        #     post_word_embeddings, _ = self.word_BiLSTM(post_embeddings[:, i, :, :])
        #     post_booled_words = self.avg_pool1d(post_word_embeddings, [x[i] for x in post_sequnce_lens])
        #     post_booled_words = post_booled_words.unsqueeze(1)
        #     if post_sentence_embeddings is None:
        #         post_sentence_embeddings = post_booled_words
        #     else:
        #         post_sentence_embeddings = torch.cat((post_sentence_embeddings, post_booled_words), 1)
        #
        # post_embeddings, _ = self.sentence_BiLSTM(post_sentence_embeddings)
        # post_doc_embedding = self.avg_pool1d(post_embeddings, post_num_sentences)
        # post_doc_embedding = torch.tanh(post_doc_embedding)#.unsqueeze(1)
        # #####################################################################################
        #
        #
        # s = Variable(torch.zeros(doc_count, 2 * self.hidden_size))
        # if self.device.type == 'cuda':
        #     s = s.cuda()
        # probs = []
        # for i in range(len(bi_sentence_embeddings[0,:,0])):
        #     #################
        #
        #     abs_index = Variable(torch.LongTensor([[i + 1] if i <= num_sentences[j] else [0] for j in range(doc_count)]))
        #     if self.device.type == 'cuda':
        #         abs_index = abs_index.cuda()
        #     abs_features = self.abs_pos_embed(abs_index).squeeze(1)
        #
        #     # rel_index = []
        #     # for x in range(len(num_sentences)):
        #     #     if i > num_sentences[x]:
        #     #         rel_in = 0
        #     #     else:
        #     #         rel_in = int(round((i + 1) * 10 / (num_sentences[x])))
        #     #     rel_index.append(rel_in)
        #     # rel_index = Variable(torch.LongTensor(rel_index))
        #     # if self.device.type == 'cuda':
        #     #     rel_index = rel_index.cuda()
        #     # rel_features = self.rel_pos_embed(rel_index).squeeze(1)
        #     ################
        #
        #     attention_and_initial_comment = self.attention_and_query(booled_sentences, post_doc_embedding)
        #     content = self.content(bi_sentence_embeddings[:,i,:])
        #     salience = self.salience(bi_sentence_embeddings[:,i,:], booled_sentences)
        #     novelty = -1 * self.novelty(bi_sentence_embeddings[:,i,:], torch.tanh(s))
        #     abs_p = self.abs_pos(abs_features)
        #     # rel_p = self.rel_pos(rel_features)
        #     prob = torch.sigmoid(content + salience + novelty + abs_p + attention_and_initial_comment + self.bias)
        #     s = s + prob * bi_sentence_embeddings[:,i,:]
        #     probs.append(prob)
        # return torch.cat(probs,1)


# class Attention(nn.Module):
#     r"""
#     Applies an attention mechanism on the query features from the decoder.
#
#     .. math::
#             \begin{array}{ll}
#             x = context*query \\
#             attn_scores = exp(x_i) / sum_j exp(x_j) \\
#             attn_out = attn * context
#             \end{array}
#
#     Args:
#         dim(int): The number of expected features in the query
#
#     Inputs: query, context
#         - **query** (batch, query_len, dimensions): tensor containing the query features from the decoder.
#         - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
#
#     Outputs: query, attn
#         - **query** (batch, query_len, dimensions): tensor containing the attended query features from the decoder.
#         - **attn** (batch, query_len, input_len): tensor containing attention weights.
#
#     Attributes:
#         mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
#
#     """
#
#     def __init__(self, attn_dropout):
#         super(Attention, self).__init__()
#         self.mask = None
#         self.dropout = nn.Dropout(attn_dropout)
#
#     def set_mask(self, mask):
#         """
#         Sets indices to be masked
#
#         Args:
#             mask (torch.Tensor): tensor containing indices to be masked
#         """
#         self.mask = mask
#
#     """
#         - query   (batch, query_len, dimensions): tensor containing the query features from the decoder.
#         - context (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
#     """
#
#     def forward(self, query, context):
#         batch_size = query.size(0)
#         dim = query.size(2)
#         in_len = context.size(1)
#         # (batch, query_len, dim) * (batch, in_len, dim) -> (batch, query_len, in_len)
#         attn = torch.bmm(query, context.transpose(1, 2))
#         if self.mask is not None:
#             attn.data.masked_fill_(self.mask, -float('inf'))
#         attn_scores = F.softmax(attn.view(-1, in_len), dim=1).view(batch_size, -1, in_len)
#         attn_scores = self.dropout(attn_scores)
#
#         # (batch, query_len, in_len) * (batch, in_len, dim) -> (batch, query_len, dim)
#         attn_out = torch.bmm(attn_scores, context)
#
#         return attn_out, attn_scores
#
#
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, q, k, v, mask=None):
#
#         attn = torch.bmm(q, k.transpose(1, 2))
#         attn = attn / self.temperature
#
#         if mask is not None:
#             attn = attn.masked_fill(mask, -np.inf)
#
#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, v)
#
#         return output, attn