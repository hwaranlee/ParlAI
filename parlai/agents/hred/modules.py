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


class PositionalEncoding(nn.Module):
  r"""Inject some information about the relative or absolute position of the tokens
      in the sequence. The positional encodings have the same dimension as
      the embeddings, so that the two can be summed. Here, we use sine and cosine
      functions of different frequencies.
  .. math::
      \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
      \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
      \text{where pos is the word position and i is the embed idx)
  Args:
      d_model: the embed dim (required).
      dropout: the dropout value (default=0.1).
      max_len: the max. length of the incoming sequence (default=5000).
  Examples:
      >>> pos_encoder = PositionalEncoding(d_model)
  """

  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    r"""Inputs of forward function
    Args:
        x: the sequence fed to the positional encoder model (required).
    Shape:
        x: [sequence length, batch size, embed dim]
        output: [sequence length, batch size, embed dim]
    Examples:
        >>> output = pos_encoder(x)
    """

    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class Maxout(nn.Module):
  def __init__(self, d_in, d_out, pool_size):
    super().__init__()
    self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
    self.lin = nn.Linear(d_in, d_out * pool_size, bias=False)

  def forward(self, inputs):
    shape = list(inputs.size())
    shape[-1] = self.d_out
    shape.append(self.pool_size)
    max_dim = len(shape) - 1
    out = self.lin(inputs)
    m, i = out.view(*shape).max(max_dim)
    return m


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

  RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU,
              'lstm': nn.LSTM, 'transformer': nn.Transformer}

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

    # Mechanism-aware
    self.num_mechanisms = opt['num_mechanisms']
    self.mechanism_size = opt['mechanism_size']

    self.max_out = Maxout(
        self.emb_size,
        self.emb_size, 2)
    self.w_t = nn.Linear(
        self.emb_size, self.mechanism_size, bias=False)
    self.softmax = nn.Softmax(dim=1)

    if self.mechanism_size == self.emb_size:
      self.to_emb = lambda x: x
    else:
      self.to_emb = nn.Linear(self.mechanism_size, self.emb_size)

    self.merge = nn.Linear(self.emb_size, 1)

    mechanisms = torch.Tensor(
        self.mechanism_size, self.num_mechanisms)
    nn.init.uniform_(mechanisms, a=-0.2, b=0.2)
    self.mechanisms = Parameter(mechanisms)

    if type(opt['gpu']) is str:
      self.gpu = [int(index) for index in opt['gpu'].split(',')]
    else:
      self.gpu = [opt['gpu']]

    # set up tensors
    self.zeros_decs = {}

    rnn_class = Hred.RNN_OPTS[opt['encoder']]
    if rnn_class == nn.Transformer:
      encoder_layer = nn.TransformerEncoderLayer(
          opt['embeddingsize'], opt['nhead'],
          dim_feedforward=opt['hiddensize'], dropout=opt['dropout'])
      self.encoder = nn.TransformerEncoder(
          encoder_layer, num_layers=opt['numlayers'])
    else:
      self.encoder = rnn_class(opt['embeddingsize'], opt['hiddensize'],
                               opt['numlayers'],
                               bidirectional=opt['bi_encoder'],
                               dropout=opt['dropout'])

    if opt['context'] != 'same':
      rnn_class = Hred.RNN_OPTS[opt['context']]

    dec_isz = opt['embeddingsize']

    if rnn_class == nn.Transformer:
      decoder_layer = nn.TransformerDecoderLayer(
          opt['embeddingsize'], opt['nhead'],
          dim_feedforward=opt['hiddensize'],
          dropout=opt['dropout'])
      self.context = nn.TransformerDecoder(
          decoder_layer, num_layers=opt['numlayers'])
    else:
      self.context = rnn_class(
          opt['numlayers'] * opt['hiddensize'], opt['contexthiddensize'],
          opt['numlayers'], dropout=opt['dropout'])

    if opt['decoder'] != 'same':
      rnn_class = Hred.RNN_OPTS[opt['decoder']]

    if rnn_class == nn.Transformer:
      decoder_layer = nn.TransformerDecoderLayer(
          opt['embeddingsize'], opt['nhead'],
          dim_feedforward=opt['hiddensize'],
          dropout=opt['dropout'])
      self.decoder = nn.TransformerDecoder(
          decoder_layer, num_layers=opt['numlayers'])
    else:
      self.decoder = rnn_class(
          dec_isz, opt['hiddensize'], opt['numlayers'], dropout=opt['dropout'])

    self.lt = nn.Embedding(
        num_features, opt['embeddingsize'], padding_idx=self.NULL_IDX)
    if opt['embed'] is not None:
      self.load_pretrained()

    self.pos_encoder = PositionalEncoding(opt['embeddingsize'], opt['dropout'])

    if 'psize' not in opt:
      opt['psize'] = opt['embeddingsize']

    if opt['embeddingsize'] == opt['psize']:
      self.o2e = lambda x: x
    else:
      self.o2e = nn.Linear(opt['embeddingsize'], opt['psize'])

    self.ch2h = nn.Linear(
        self.num_layers * opt['contexthiddensize'] + self.mechanism_size,
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

    self.generate_square_subsequent_mask = nn.Transformer().generate_square_subsequent_mask

  def load_pretrained(self):
    model = gensim.models.word2vec.Word2Vec.load(self.opt['embed']).wv
    std = model.vectors.std().item()
    word2vec_size = model.vectors.shape[1]
    n_unk = 0
    for i in range(len(self.lt.weight)):
      if i == 0:
        self.lt.weight.data[i].zero_()
      else:
        word = self.opt['dict'].vec2txt([i])

        try:
          self.lt.weight.data[i][:word2vec_size] = torch.from_numpy(
              model[word])
          self.lt.weight.data[i][word2vec_size:].normal_(0, std)
        except KeyError:
          n_unk += 1
          self.lt.weight.data[i].normal_(0, std)
    print('unk_num: {}'.format(n_unk))

  def cuda(self):
    if len(self.gpu) > 1:
      self.START = self.START.cuda(self.gpu[0])
      self.lt.cuda(self.gpu[0])
      self.pos_encoder.cuda(self.gpu[0])
      self.encoder.cuda(self.gpu[0])
      if len(self.gpu) == 4:
        self.context.cuda(self.gpu[1])
        if type(self.to_emb) is nn.Linear:
          self.to_emb.cuda(self.gpu[1])
        self.merge.cuda(self.gpu[1])
        self.ch2h.cuda(self.gpu[1])
        self.max_out.cuda(self.gpu[1])
        self.w_t.cuda(self.gpu[1])
        self.mechanisms = Parameter(self.mechanisms.cuda(self.gpu[1]))
        self.decoder.cuda(self.gpu[2])
        self.dropout.cuda(self.gpu[2])
        if type(self.o2e) is nn.Linear:
          self.o2e.cuda(self.gpu[2])
        self.e2s.cuda(self.gpu[3])
      else:
        self.context.cuda(self.gpu[0])
        if type(self.to_emb) is nn.Linear:
          self.to_emb.cuda(self.gpu[0])
        self.merge.cuda(self.gpu[0])
        self.ch2h.cuda(self.gpu[0])
        self.max_out.cuda(self.gpu[0])
        self.w_t.cuda(self.gpu[0])
        self.mechanisms = Parameter(self.mechanisms.cuda(self.gpu[0]))
        self.decoder.cuda(self.gpu[0])
        self.dropout.cuda(self.gpu[0])
        if type(self.o2e) is nn.Linear:
          self.o2e.cuda(self.gpu[0])
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
    xes = self.pos_encoder(xes)
    seqlen = len(xes)

    xlen, idx = xlen.sort(descending=True)
    zero_len = (xlen == -1).nonzero()
    # hidden = hidden.index_select(1, idx)
    if len(zero_len):
      first_zero_idx = zero_len[0].item()
      xes = xes.index_select(1, idx[:first_zero_idx])  # len, batch, embsize
      xs_selected = xs.index_select(0, idx[:first_zero_idx])
      # xes = pack_padded_sequence(
      #     xes, (xlen[:first_zero_idx] + 1).data.cpu().numpy())
      # hidden, hidden_left = hidden.split(
      #     [first_zero_idx, batchsize - first_zero_idx], 1)
    else:
      xes = xes.index_select(1, idx)
      xs_selected = xs.index_select(0, idx)
      # xes = pack_padded_sequence(xes, (xlen + 1).data.cpu().numpy())

    # self.encoder.flatten_parameters()
    # _, hidden = self.encoder(xes, hidden.contiguous())
    # len, batch, embsize
    hidden = self.encoder(xes, src_key_padding_mask=Hred.to_mask(xs_selected))

    # hidden = hidden.view(
    #     -1, self.dirs,
    #     batchsize - len(zero_len), self.hidden_size).max(1)[0]

    if len(zero_len):
      zeros = self.zeros(encoder_device)
      if list(zeros.size()) != [seqlen,
                                batchsize - first_zero_idx, self.emb_size]:
        zeros.resize_(seqlen,
                      batchsize - first_zero_idx, self.emb_size).fill_(0)
      hidden_left = Variable(zeros, requires_grad=False)
      hidden = torch.cat((hidden, hidden_left), 1)

    undo_idx = idx.clone()
    for i in range(len(idx)):
      undo_idx[idx[i]] = i

    hidden = hidden.index_select(1, undo_idx)

    return hidden

  def _context(self, context_hidden, hidden, hidden_mask, m_idx=None):
    batchsize = hidden.size(1)
    context_device = next(self.context.parameters()).get_device()

    # hidden = hidden.transpose(0, 1).contiguous().view(1, batchsize, -1)

    if len(self.gpu) > 1:
      hidden = hidden.cuda(context_device)

    if context_hidden is None:
      zeros = self.zeros(context_device)
      if list(zeros.size()) != [1, batchsize,
                                self.emb_size]:
        zeros.resize_(1,
                      batchsize, self.emb_size).fill_(0)
      context_hidden = Variable(zeros, requires_grad=False)

    return self.context(context_hidden, hidden,
                        memory_key_padding_mask=hidden_mask)

  def _decode(self, batchsize, ys, hidden, hidden_mask):
    decoder_device = next(self.decoder.parameters()).get_device()
    lt_device = next(self.lt.parameters()).get_device()
    # update the model based on the labels
    if len(self.gpu) > 1:
      hidden = hidden.cuda(decoder_device)

    mask = None
    if ys is None:
      raise Exception('Hope this is not used.')
      # done = [False] * batchsize
      # total_done = 0
      # max_len = 0
      # while total_done < batchsize and max_len < self.longest_label:
      #   # keep producing tokens until we hit END or max length for each
      #   current_output, hidden = self.decoder(
      #       output, hidden,
      #       tgt_mask=self.generate_square_subsequent_mask(max_len + 1),
      #       tgt_key_padding_mask=mask,
      #       memory_key_padding_mask=hidden_mask)
      #   hidden = torch.cat((hidden, output))
      #   pred, score = self.hidden_to_idx(current_output, dropout=self.training)
      #   preds.append(pred)
      #   scores.append(score)
      #   mask = Hred.to_mask(pred.unsqueeze(1))
      #   if len(self.gpu) > 1:
      #     pred = pred.cuda(lt_device)
      #
      #   output = self.lt(pred).unsqueeze(0)
      #
      #   if len(self.gpu) > 1:
      #     output = output.cuda(decoder_device)
      #
      #   max_len += 1
      #   for b in range(batchsize):
      #     if not done[b]:
      #       # only add more tokens for examples that aren't done yet
      #       if pred.data[b] == self.END:
      #         # if we produced END, we're done
      #         done[b] = True
      #         total_done += 1
    else:
      if len(self.gpu) > 1:
        ys = ys.cuda(lt_device)

      ys = torch.cat((self.START.expand(ys.size(0), 1), ys[:, :-1]), dim=1)
      yes = self.lt(ys).transpose(0, 1)
      yes = self.pos_encoder(yes)
      tgt_key_padding_mask = Hred.to_mask(ys)

      if len(self.gpu) > 1:
        yes = yes.cuda(decoder_device)
        hidden_mask = hidden_mask.cuda(decoder_device)
        tgt_key_padding_mask = tgt_key_padding_mask.cuda(decoder_device)

      # keep track of longest label we've ever seen
      self.longest_label = max(self.longest_label, ys.size(1))

      tgt_mask = self.generate_square_subsequent_mask(yes.size(0))
      if len(self.gpu) > 1:
        tgt_mask = tgt_mask.cuda(decoder_device)
      output = self.decoder(
          yes, hidden,
          tgt_mask=tgt_mask,
          tgt_key_padding_mask=tgt_key_padding_mask,
          memory_key_padding_mask=hidden_mask)
      preds, scores = self.hiddens_to_idx(
          output,
          dropout=self.training)

    return scores, preds

  @staticmethod
  def to_mask(input):  # Assume padding is zero.
    return input == 0

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

  def hiddens_to_idx(self, hidden, dropout=False):
    """Convert hidden state vectors into indices into the dictionary."""
    e2s_device = next(self.e2s.parameters()).get_device()
    scores = self.o2e(hidden)  # seq, batch, psize
    if len(self.gpu) > 1:
      scores = scores.cuda(e2s_device)
    scores = self.e2s(scores)
    scores = F.log_softmax(scores, 2)
    _max_score, idx = scores.max(2)
    return idx, scores

  @staticmethod
  def contains_nan(x):
    return math.isnan(x.sum().item())

  def forward(self, xses, dropout, xlen_ts, ys, m_idx=None):
    batchsize = len(xses[0])

    context_hidden = None
    for idx in range(0, len(xses)):
      hidden = self._encode(xses[idx], xlen_ts[idx], dropout)
      mask = Hred.to_mask(xses[idx])
      context_hidden = self._context(
          context_hidden, hidden, mask)

    transposed_context_hidden = context_hidden.transpose(0, 1)

    if m_idx is None:
      alphas = self.softmax(self.merge(transposed_context_hidden))
      merged = (alphas.transpose(1, 2) @ transposed_context_hidden).squeeze(1)

      t = self.max_out(merged)
      w = self.w_t(t)
      g = w @ self.mechanisms
      p_m = self.softmax(g)
      m = p_m @ self.mechanisms.t()
    else:
      m = self.mechanisms[:, m_idx, None].t()

    m = self.to_emb(m).unsqueeze(0)

    # hidden = self.tanh(self.ch2h(torch.cat((hidden, m), dim=1)).view(
    #     batchsize, self.num_layers, -1).transpose(0, 1))

    hidden = torch.cat((hidden, context_hidden, m), dim=0)
    mask = torch.cat((mask, torch.BoolTensor(batchsize, 2).new_full(
        (batchsize, 2), False
    ).cuda(mask.get_device())), dim=1)

    scores, preds = self._decode(
        batchsize, ys, hidden, mask)

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
