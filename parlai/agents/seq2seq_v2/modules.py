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

    def __init__(self, opt, num_features, start_tensor, padding_idx=0, longest_label=1):
        super().__init__()
        self.opt = opt

        self.longest_label = longest_label
        
        # we use START markers to start our output
        self.START_TENSOR = start_tensor
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
        if self.opt['bi_encoder']:   
            self.zeros = torch.zeros(2 * self.num_layers, 1, hsz)
        else:
            self.zeros = torch.zeros(self.num_layers, 1, hsz)
            
        self.zeros_dec = torch.zeros(self.num_layers, 1, hsz)

        self.xs = torch.LongTensor(1, 1)
        self.ys = torch.LongTensor(1, 1)
        self.cands = torch.LongTensor(1, 1, 1)
        self.cand_scores = torch.FloatTensor(1)
        self.cand_lengths = torch.LongTensor(1)

        rnn_class = Seq2seq.RNN_OPTS[opt['encoder']]
        self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        if opt['decoder'] != 'same':
            rnn_class = Seq2seq.RNN_OPTS[opt['decoder']]

        dec_isz = opt['embeddingsize'] + opt['hiddensize']

        if opt['bi_encoder']:
            dec_isz += opt['hiddensize']

        self.decoder = rnn_class(dec_isz, opt['hiddensize'], opt['numlayers'], bidirectional=opt['bi_encoder'], dropout=opt['dropout'])

        self.lt = nn.Embedding(num_features, opt['embeddingsize'], padding_idx=self.NULL_IDX)

        self.criterion = nn.NLLLoss(size_average=False, ignore_index=0)
        self.h2o = nn.Linear(opt['hiddensize'], num_features)
        self.dropout = nn.Dropout(opt['dropout'])

        optim_class = Seq2seq.OPTIM_OPTS[opt['optimizer']]
        kwargs = {'lr': opt['learning_rate']}
        if opt['optimizer'] == 'sgd':
            kwargs['momentum'] = 0.95
            kwargs['nesterov'] = True
        self.optimizer = optim_class(self.parameters(), **kwargs)

        self.use_attention = False

        self.episode_concat = opt['episode_concat']
        self.training = True
        self.generating = False
        self.local_human = False
        
        if opt.get('max_seq_len') is not None:
            self.max_seq_len = opt['max_seq_len']
        else:
            self.max_seq_len = opt['max_seq_len'] = 50

    def _encode(self, xs, xlen, dropout=False, packed=True):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).transpose(0, 1)
        # if dropout:
        #    xes = self.dropout(xes)
        
        # initial hidden 
        if self.zeros.size(1) != batchsize:
            if self.opt['bi_encoder']:   
                self.zeros.resize_(2 * self.num_layers, batchsize, self.hidden_size).fill_(0) 
            else:
                self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0) 
            
        h0 = Variable(self.zeros.fill_(0))

        # forward
        if packed:
            xes = torch.nn.utils.rnn.pack_padded_sequence(xes, xlen)

        if type(self.encoder) == nn.LSTM:
            encoder_output, _ = self.encoder(xes, (h0, h0))  # # Note : we can put None instead of (h0, h0)
        else:
            encoder_output, _ = self.encoder(xes, h0)
        
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

    def zero_grad(self):
        """Zero out optimizers."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()
            
    def _get_context(self, batchsize, xlen_t, encoder_output):
        " return initial hidden of decoder and encoder context (last_state)"
        
        # # The initial of decoder is the hidden (last states) of encoder --> put zero!       
        if self.zeros_dec.size(1) != batchsize:
            self.zeros_dec.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        hidden = Variable(self.zeros_dec.fill_(0))
        
        last_state = None
        if not self.use_attention:
            last_state = torch.gather(encoder_output, 1, xlen_t.view(-1, 1, 1).expand(encoder_output.size(0), 1, encoder_output.size(2)))
            if self.opt['bi_encoder']:
#                last_state = torch.cat((encoder_output[:,0,:self.hidden_size], last_state[:,0,self.hidden_size:]),1)        
                last_state = torch.cat((encoder_output[:, 0, self.hidden_size:], last_state[:, 0, :self.hidden_size]), 1)        
        
        return hidden, last_state

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
            
            print(output.size())
            print(hidden.size())
            output, hidden = self.decoder(output, hidden)           
            preds, scores = self.hidden_to_idx(output, dropout=self.training)
            y = ys.select(1, i)
            loss += self.criterion(scores, y)  # not averaged
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            dec_xes = self.lt(y).unsqueeze(0)
            
            # TODO: overhead!
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([preds.data[b]])
                output_lines[b].append(token)
        
        if self.training:
            self.loss = loss.data[0] / sum(ylen)  # consider non-NULL
            self.ndata += batchsize
        else:
            self.loss_valid += loss.data[0]  # consider non-NULL / accumulate!
            self.ndata_valid += sum(ylen)          

        return loss, output_lines
    
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

    def forward(self, xs, xlen, dropout, batchsize, xlen_t, ys, ylen):
        encoder_output = self._encode(xs, xlen, dropout)
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        output_lines = None

        loss, output_lines = self._decode_and_train(batchsize, dec_xes, xlen_t, xs, ys, ylen, encoder_output)

        return loss, output_lines

