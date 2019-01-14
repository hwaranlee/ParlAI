# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import gensim

import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable


class Hred(nn.Module):
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
    self.truncate = opt['truncate']
    self.attention = opt['attention']
    self.dirs = 2 if opt['bi_encoder'] else 1
    if type(opt['gpu']) is str:
      self.gpu = [int(index) for index in opt['gpu'].split(',')]
    else:
      self.gpu = [opt['gpu']]

    # set up tensors
    self.zeros_decs = {}

    rnn_class = Hred.RNN_OPTS[opt['encoder']]
    self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'],
                             opt['numlayers'], bidirectional=opt['bi_encoder'],
                             dropout=opt['dropout'])

    if opt['context'] != 'same':
      rnn_class = Hred.RNN_OPTS[opt['context']]

    dec_isz = opt['embeddingsize']

    self.context = rnn_class(
        opt['numlayers'] * opt['hiddensize'], opt['contexthiddensize'],
        opt['numlayers'], dropout=opt['dropout'])

    if opt['decoder'] != 'same':
      rnn_class = Hred.RNN_OPTS[opt['decoder']]

    self.decoder = rnn_class(
        dec_isz, opt['hiddensize'], opt['numlayers'], dropout=opt['dropout'])

    self.lt = nn.Embedding(
        num_features, opt['embeddingsize'], padding_idx=self.NULL_IDX)
    if opt['embed'] is not None:
      self.load_pretrained()

    if 'psize' not in opt:
      opt['psize'] = opt['embeddingsize']

    if opt['hiddensize'] == opt['psize']:
      self.o2e = lambda x: x
    else:
      self.o2e = nn.Linear(opt['hiddensize'], opt['psize'])

    self.ch2h = nn.Linear(
        self.num_layers * opt['contexthiddensize'],
        self.num_layers * opt['hiddensize'])
    self.tanh = nn.Tanh()

    share_output = opt['lookuptable'] in ['dec_out', 'all'] and \
        opt['psize'] == opt['embeddingsize']
    shared_weight = self.lt.weight if share_output else None
    self.e2s = Linear(opt['psize'], num_features,
                      bias=False, shared_weight=shared_weight)
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
          n_unk += 1
          self.lt.weight.data[i].normal_(0, std)
    print('unk_num: {}'.format(n_unk))

  def cuda(self):
    if len(self.gpu) > 1:
      self.START = self.START.cuda(self.gpu[0])
      self.lt.cuda(self.gpu[0])
      self.encoder.cuda(self.gpu[0])
      if len(self.gpu) == 4:
        self.context.cuda(self.gpu[1])
        self.ch2h.cuda(self.gpu[1])
        self.decoder.cuda(self.gpu[2])
        self.dropout.cuda(self.gpu[2])
        if type(self.o2e) is nn.Linear:
          self.o2e.cuda(self.gpu[2])
        self.e2s.cuda(self.gpu[3])
      else:
        self.context.cuda(self.gpu[0])
        self.ch2h.cuda(self.gpu[0])
        self.decoder.cuda(self.gpu[1])
        self.dropout.cuda(self.gpu[1])
        if type(self.o2e) is nn.Linear:
          self.o2e.cuda(self.gpu[1])
        self.e2s.cuda(self.gpu[-1])
    else:
      super().cuda()

  def zeros(self, device_id):
    # if device_id in self.zeros_decs:
    #   ret = self.zeros_decs[device_id]
    # else:
    ret = torch.zeros(1, 1, 1).cuda(device_id)
    # self.zeros_decs[device_id] = ret

    return ret

  def _encode(self, xs, xlen, dropout=False):
    """Call encoder and return output and hidden states."""
    encoder_device = next(self.encoder.parameters()).get_device()
    batchsize = len(xs)

    # first encode context
    xes = self.lt(xs).transpose(0, 1)

    zeros = self.zeros(encoder_device)
    if list(zeros.size()) != [self.dirs * self.num_layers, batchsize, self.hidden_size]:
      zeros.resize_(self.dirs * self.num_layers,
                    batchsize, self.hidden_size).fill_(0)
    hidden = Variable(zeros, requires_grad=False)

    xlen, idx = xlen.sort(descending=True)
    zero_len = (xlen == -1).nonzero()
    hidden = hidden.index_select(1, idx)
    if len(zero_len):
      first_zero_idx = zero_len[0].item()
      xes = xes.index_select(1, idx[:first_zero_idx])
      xes = pack_padded_sequence(
          xes, (xlen[:first_zero_idx] + 1).data.cpu().numpy())
      hidden, hidden_left = hidden.split(
          [first_zero_idx, batchsize - first_zero_idx], 1)
    else:
      xes = xes.index_select(1, idx)
      xes = pack_padded_sequence(xes, (xlen + 1).data.cpu().numpy())

    # self.encoder.flatten_parameters()
    _, hidden = self.encoder(xes, hidden.contiguous())

    hidden = hidden.view(
        -1, self.dirs,
        batchsize - len(zero_len), self.hidden_size).max(1)[0]

    if len(zero_len):
      hidden = torch.cat((hidden, hidden_left), 1)

    undo_idx = idx.clone()
    for i in range(len(idx)):
      undo_idx[idx[i]] = i

    hidden = hidden.index_select(1, undo_idx)

    return hidden

  def _context(self, hidden, context_hidden):
    batchsize = hidden.size(1)
    context_device = next(self.context.parameters()).get_device()

    hidden = hidden.transpose(0, 1).contiguous().view(1, batchsize, -1)

    if len(self.gpu) > 1:
      hidden = hidden.cuda(context_device)

    if context_hidden is None:
      zeros = self.zeros(context_device)
      if list(zeros.size()) != [self.dirs * self.num_layers, batchsize,
                                self.opt['contexthiddensize']]:
        zeros.resize_(self.dirs * self.num_layers,
                      batchsize, self.opt['contexthiddensize']).fill_(0)
      context_hidden = Variable(zeros, requires_grad=False)

    _, context_hidden = self.context(hidden, context_hidden)
    hidden = context_hidden.transpose(0, 1).contiguous().view(batchsize, -1)

    hidden = self.tanh(self.ch2h(hidden).view(
        batchsize, self.num_layers, -1).transpose(0, 1))

    return hidden, context_hidden

  def _decode(self, batchsize, output, ys, hidden):
    decoder_device = next(self.decoder.parameters()).get_device()
    lt_device = next(self.lt.parameters()).get_device()
    # update the model based on the labels
    scores = []
    if len(self.gpu) > 1:
      output = output.cuda(decoder_device)
      hidden = hidden.cuda(decoder_device)

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

        if len(self.gpu) > 1:
          pred = pred.cuda(lt_device)

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
        if len(self.gpu) > 1:
          y = y.cuda(lt_device)

        output = self.lt(y).unsqueeze(0)

        if len(self.gpu) > 1:
          output = output.cuda(decoder_device)
    preds = torch.stack(preds, 1)

    return scores, preds

  def hidden_to_idx(self, hidden, dropout=False):
    """Convert hidden state vectors into indices into the dictionary."""
    e2s_device = next(self.e2s.parameters()).get_device()
    if hidden.size(0) > 1:
      raise RuntimeError('bad dimensions of tensor:', hidden)
    hidden = hidden.squeeze(0)
    if dropout:
      hidden = self.dropout(hidden)  # dropout over the last hidden
    scores = self.o2e(hidden)
    if len(self.gpu) > 2:
      scores = scores.cuda(e2s_device)
    scores = self.e2s(scores)
    scores = F.log_softmax(scores, 1)
    _max_score, idx = scores.max(1)
    return idx, scores

  def forward(self, xses, dropout, xlen_ts, ys):
    batchsize = len(xses[0])

    context_hidden = None
    for idx in range(0, len(xses)):
      hidden = self._encode(xses[idx], xlen_ts[idx], dropout)
      hidden, context_hidden = self._context(hidden, context_hidden)

    x = Variable(self.START, requires_grad=False)
    xe = self.lt(x).unsqueeze(1)
    dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

    scores, preds = self._decode(batchsize, dec_xes, ys, hidden)

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
