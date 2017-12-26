# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
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

    def __init__(self, opt, num_features,
                 padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()
        self.opt = opt

        self.rank = opt['rank_candidates']
        self.lm = opt['language_model']
        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2seq.RNN_OPTS[opt['encoder']]
        self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        if opt['decoder'] != 'same':
            rnn_class = Seq2seq.RNN_OPTS[opt['decoder']]

        dec_isz = opt['embeddingsize'] + opt['hiddensize']

        if opt['bi_encoder']:
            dec_isz += opt['hiddensize']

        self.decoder = rnn_class(dec_isz, opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        self.lt = nn.Embedding(len(self.dict), opt['embeddingsize'], padding_idx=self.NULL_IDX)

        self.h2o = nn.Linear(opt['hiddensize'], len(self.dict))
        self.dropout = nn.Dropout(opt['dropout'])

        self.use_attention = False
        self.attn = None

        self.episode_concat = opt['episode_concat']
        self.training = True
        self.generating = False
        self.local_human = False
        
        if opt.get('max_seq_len') is not None:
            self.max_seq_len = opt['max_seq_len']
        else:
            self.max_seq_len = opt['max_seq_len'] = 50
        self.reset()

    def _encode(self, xs, xlen, dropout=False, packed=True):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).transpose(0, 1)
        #if dropout:
        #    xes = self.dropout(xes)
        
        # initial hidden 
        if self.zeros.size(1) != batchsize:
            if self.opt['bi_encoder']:   
                self.zeros.resize_(2*self.num_layers, batchsize, self.hidden_size).fill_(0) 
            else:
                self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0) 
            
        h0 = Variable(self.zeros.fill_(0))
        
        # forward
        if packed:
            xes = torch.nn.utils.rnn.pack_padded_sequence(xes, xlen)
                
        if type(self.encoder) == nn.LSTM:
            encoder_output, _ = self.encoder(xes, (h0, h0)) ## Note : we can put None instead of (h0, h0)
        else:
            encoder_output, _ = self.encoder(xes, h0)
        
        if packed:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)
            
        encoder_output = encoder_output.transpose(0, 1) #batch first
        
        """
        if self.use_attention:
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)
        """
                
        return encoder_output


    def _decode_and_train(self, batchsize, dec_xes, xlen_t, xs, ys, ylen, encoder_output):
        # update the model based on the labels
        self.zero_grad()
        loss = 0
        
        output_lines = [[] for _ in range(batchsize)]
        
        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))
        
        hidden, last_state = self._get_context(batchsize, xlen_t, encoder_output)

        for i in range(ys.size(1)):
            if self.use_attention:
                output = self._apply_attention(dec_xes, encoder_output, hidden[-1], xs)
            else:                
                output = torch.cat((dec_xes, last_state.unsqueeze(0)), 2)
            
            output, hidden = self.decoder(output, hidden)           
            preds, scores = self.hidden_to_idx(output, dropout=self.training)
            y = ys.select(1, i)
            loss += self.criterion(scores, y) #not averaged
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            dec_xes = self.lt(y).unsqueeze(0)
            
            # TODO: overhead!
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([preds.data[b]])
                output_lines[b].append(token)
        
        if self.training:
            self.loss = loss.data[0]/sum(ylen) # consider non-NULL
            self.ndata += batchsize
        else:
            self.loss_valid += loss.data[0] # consider non-NULL / accumulate!
            self.ndata_valid += sum(ylen)          

        return loss, output_lines


    def forward(self, xs, xlen, dropout, batchsize, xlen_t, ys, ylen):
        encoder_output = self._encode(xs, xlen, dropout)
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        output_lines = None

        loss, output_lines = self._decode_and_train(batchsize, dec_xes, xlen_t, xs, ys, ylen, encoder_output)

        return loss, output_lines

