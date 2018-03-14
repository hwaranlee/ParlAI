# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import gensim
from torch import optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


class Seq2seq(nn.Module):
    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }
    
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, num_features, start_idx, padding_idx=0, longest_label=1):
        super().__init__()
        self.opt = opt

        self.longest_label = longest_label
        
        self.register_buffer('START', torch.LongTensor(start_idx))
        # get index of null token from dictionary (probably 0)
        self.NULL_IDX = padding_idx
        self.END = 1
        
        # store important params directly
        hsz = opt['hiddensize']
        emb = opt['embeddingsize']
        self.hidden_size = hsz
        self.emb_size = emb
        self.num_layers = opt['numlayers']
        self.learning_rate = opt['learning_rate']
        self.rank = opt['rank_candidates']
        self.truncate = opt['truncate']
        self.attention = opt['attention']
        self.dirs = 2 if opt['bi_encoder'] else 1
        self.split_gpus = opt['split_gpus']
        
        # set up tensors
        self.zeros_decs = {}

        rnn_class = Seq2seq.RNN_OPTS[opt['encoder']]
        self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        if opt['decoder'] != 'same':
            rnn_class = Seq2seq.RNN_OPTS[opt['decoder']]

        dec_isz = opt['embeddingsize']

        self.decoder = rnn_class(dec_isz, opt['hiddensize'], opt['numlayers'], dropout=opt['dropout'])

        self.lt = nn.Embedding(num_features, opt['embeddingsize'], padding_idx=self.NULL_IDX)
        if opt['embed'] is not None:
            self.load_pretrained()

        if opt['hiddensize'] == opt['embeddingsize']:
            self.o2e = lambda x: x
        else:
            self.o2e = nn.Linear(opt['hiddensize'], opt['embeddingsize'])

        share_output = opt['lookuptable'] in ['dec_out', 'all']
        shared_weight = self.lt.weight if share_output else None
        self.e2s = Linear(opt['embeddingsize'], num_features, bias=False, shared_weight=shared_weight)
        self.dropout = nn.Dropout(opt['dropout'])

        self.use_attention = False

        self.episode_concat = opt['episode_concat']
        self.training = True
        self.generating = False
        self.local_human = False
        
        if opt.get('max_seq_len') is not None:
            self.max_seq_len = opt['max_seq_len']
        else:
            self.max_seq_len = opt['max_seq_len'] = 50

    def load_pretrained(self):
        model = gensim.models.word2vec.Word2Vec.load(self.opt['embed']).wv
        std = model.vectors.std().item()
        n_unk = 0
        for i in range(len(self.lt.weight)):
            if i == 0:
                self.lt.weight.data[i].zero_()
            else:
                word = self.opt['dict'].vec2txt([i])

                try:
                    self.lt.weight.data[i] = torch.from_numpy(model[word])
                except KeyError:
                    print(word)
                    n_unk += 1
                    self.lt.weight.data[i].normal_(0, std)
        print('unk_num: {}'.format(n_unk))

    def cuda(self):
        if self.split_gpus:
            self.START = self.START.cuda(0)
            self.lt.cuda(0)
            self.encoder.cuda(0)
            self.decoder.cuda(1)
            self.o2e.cuda(1)
            self.e2s.cuda(1)
            self.dropout.cuda(1)
        else:
            super().cuda()

    def zeros(self, device_id):
        if device_id in self.zeros_decs:
            ret = self.zeros_decs[device_id]
        else:
            ret = torch.zeros(1, 1, 1).cuda(device_id)
            self.zeros_decs[device_id] = ret

        return ret

    def _encode(self, xs, xlen, dropout=False, packed=True):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).transpose(0, 1)
        # if dropout:
        #    xes = self.dropout(xes)
        
        # forward
        if packed:
            xes = torch.nn.utils.rnn.pack_padded_sequence(xes, (xlen + 1).data.cpu().numpy()) 

        zeros = self.zeros(xs.get_device())
        if list(zeros.size()) != [self.dirs * self.num_layers, batchsize, self.hidden_size]:
            zeros.resize_(self.dirs * self.num_layers, batchsize, self.hidden_size).fill_(0)
        h0 = Variable(zeros, requires_grad=False)

        # self.encoder.flatten_parameters()
        encoder_output, hidden = self.encoder(xes, h0)
        hidden = hidden.view(-1, self.dirs, batchsize, self.hidden_size).max(1)[0]
        
        if packed:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)

        encoder_output = encoder_output.transpose(0, 1)  # batch first
        
        """
        if self.use_attention:
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)
        """
                
        return encoder_output, hidden

    def _decode_and_train(self, batchsize, output, xlen_t, xs, ys, hidden):
        # update the model based on the labels
        scores = []
        if self.split_gpus:
            output = output.cuda(1)
            hidden = hidden.cuda(1)

        preds = []
        if ys is None:
            done = [False] * batchsize
            total_done = 0
            max_len = 0
            while total_done < batchsize and max_len < self.longest_label:
                # keep producing tokens until we hit END or max length for each
                output, hidden = self.decoder(output, hidden)
                pred, score = self.hidden_to_idx(output, dropout=self.training)
                preds.append(pred)
                scores.append(score)
    
                if self.split_gpus:
                    pred = pred.cuda(0)

                output = self.lt(pred).unsqueeze(0)
                
                max_len += 1
                for b in range(batchsize):
                    if not done[b]:
                        # only add more tokens for examples that aren't done yet
                        if pred.data[b] == self.END:
                            # if we produced END, we're done
                            done[b] = True
                            total_done += 1
        else:
            # keep track of longest label we've ever seen
            self.longest_label = max(self.longest_label, ys.size(1))

            for i in range(ys.size(1)):
                output, hidden = self.decoder(output, hidden)           
                pred, score = self.hidden_to_idx(output, dropout=self.training)
                preds.append(pred)
                scores.append(score)
                y = ys.select(1, i)
                if self.split_gpus:
                    y = y.cuda(0)
    
                output = self.lt(y).unsqueeze(0)
    
                if self.split_gpus:
                    output = output.cuda(1)
        preds = torch.stack(preds, 1)

        return scores, preds
    
    def hidden_to_idx(self, hidden, dropout=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        if dropout:
            hidden = self.dropout(hidden)  # dropout over the last hidden
        scores = self.o2e(hidden)
        scores = self.e2s(scores)
        scores = F.log_softmax(scores, 1)
        _max_score, idx = scores.max(1)
        return idx, scores

    def forward(self, xs, dropout, xlen_t, ys):
        batchsize = len(xs)
        encoder_output, hidden = self._encode(xs, xlen_t, dropout)
        x = Variable(self.START, requires_grad=False)
        xe = self.lt(x).unsqueeze(1)
        dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        output_lines = None

        scores, preds = self._decode_and_train(batchsize, dec_xes, xlen_t, xs, ys, hidden)

        return scores, preds


class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
