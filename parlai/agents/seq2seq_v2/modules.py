# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
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
        
        # store important params directly
        hsz = opt['hiddensize']
        emb = opt['embeddingsize']
        self.hidden_size = hsz
        self.emb_size = emb
        self.num_layers = opt['numlayers']
        self.learning_rate = opt['learning_rate']
        self.rank = opt['rank_candidates']
        self.longest_label = 1
        self.truncate = opt['truncate']
        self.attention = opt['attention']
        
        # set up tensors
        self.zeros_decs = {}

        rnn_class = Seq2seq.RNN_OPTS[opt['encoder']]
        self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        if opt['decoder'] != 'same':
            rnn_class = Seq2seq.RNN_OPTS[opt['decoder']]

        dec_isz = opt['embeddingsize'] + opt['hiddensize']

        if opt['bi_encoder']:
            dec_isz += opt['hiddensize']

        self.decoder = rnn_class(dec_isz, opt['hiddensize'], opt['numlayers'], dropout=opt['dropout'])

        self.lt = nn.Embedding(num_features, opt['embeddingsize'], padding_idx=self.NULL_IDX)

        self.h2o = nn.Linear(opt['hiddensize'], num_features)
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

        self.encoder.flatten_parameters()
        encoder_output, _ = self.encoder(xes)
        
        if packed:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)

        encoder_output = encoder_output.transpose(0, 1)  # batch first
        
        """
        if self.use_attention:
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)
        """
                
        return encoder_output

    def _get_context(self, batchsize, xlen_t, encoder_output):
        " return initial hidden of decoder and encoder context (last_state)"
        
        # # The initial of decoder is the hidden (last states) of encoder --> put zero!       
        zeros = self.zeros(encoder_output.get_device())
        if zeros.size(1) != batchsize:
            zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        hidden = Variable(zeros.fill_(0))
        
        last_state = None
        if not self.use_attention:
            last_state = torch.gather(encoder_output, 1, xlen_t.view(-1, 1, 1).expand(encoder_output.size(0), 1, encoder_output.size(2)))
            if self.opt['bi_encoder']:
                last_state = torch.cat((encoder_output[:, 0, self.hidden_size:], last_state[:, 0, :self.hidden_size]), 1)        
        
        return hidden, last_state

    def _decode_and_train(self, batchsize, dec_xes, xlen_t, xs, ys, encoder_output):
        # update the model based on the labels
        scores = []
        
        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))

        hidden, last_state = self._get_context(batchsize, xlen_t, encoder_output)

        for i in range(ys.size(1)):
            if self.use_attention:
                output = self._apply_attention(dec_xes, encoder_output, hidden[-1], xs)
            else:                
                output = torch.cat((dec_xes, last_state.unsqueeze(0)), 2)
            
            self.decoder.flatten_parameters()
            output, hidden = self.decoder(output, hidden)           
            preds, score = self.hidden_to_idx(output, dropout=self.training)
            scores.append(score)
            y = ys.select(1, i)
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            dec_xes = self.lt(y).unsqueeze(0)
        
        return scores, preds
    
    def hidden_to_idx(self, hidden, dropout=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        if dropout:
            hidden = self.dropout(hidden)  # dropout over the last hidden
        scores = self.h2o(hidden)
        scores = F.log_softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def forward(self, xs, dropout, xlen_t, ys):
        batchsize = len(xs)
        encoder_output = self._encode(xs, xlen_t, dropout)
        x = Variable(self.START, requires_grad=False)
        xe = self.lt(x).unsqueeze(1)
        dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        output_lines = None

        scores, preds = self._decode_and_train(batchsize, dec_xes, xlen_t, xs, ys, encoder_output)

        return scores, preds

