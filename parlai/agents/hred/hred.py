# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import str2class
from .beam import Beam
from .modules import Hred
from ..seq2seq_v2.modules import Seq2seq
from datetime import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import importlib

import os
import random
import math
import re


class HredAgent(Agent):
  """Agent which takes an input sequence and produces an output sequence.

  For more information, see Sequence to Sequence Learning with Neural
  Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
  """
  ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU,
              'lstm': nn.LSTM, 'transformer': nn.Transformer}

  @staticmethod
  def add_cmdline_args(argparser):
    """Add command-line arguments specifically for this agent."""
    DictionaryAgent.add_cmdline_args(argparser)
    agent = argparser.add_argument_group('HRED Arguments')
    agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                       help='size of the hidden layers')
    agent.add_argument('-chs', '--contexthiddensize', type=int, default=128,
                       help='size of the hidden layers')
    agent.add_argument('-emb', '--embeddingsize', type=int, default=128,
                       help='size of the token embeddings')
    agent.add_argument('-nl', '--numlayers', type=int, default=2,
                       help='number of hidden layers')
    agent.add_argument('-lr', '--learning_rate', type=float, default=0.5,
                       help='learning rate')
    agent.add_argument('-wd', '--weight_decay', type=float, default=0,
                       help='weight decay')
    agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                       help='dropout rate')
    agent.add_argument('-att', '--attention', default=False, type='bool',
                       help='if True, use attention')
    agent.add_argument('-attType', '--attn-type', default='general',
                       choices=['general', 'concat', 'dot'],
                       help='general=bilinear dotproduct, ' +
                       'concat=bahdanau\'s implemenation')
    agent.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable GPUs even if available')
    agent.add_argument('--gpu', type=str, default='-1',
                       help='which GPU device to use')
    agent.add_argument('-rc', '--rank-candidates', type='bool',
                       default=False,
                       help='rank candidates if available. this is done by'
                            ' computing the mean score per token for each '
                            'candidate and selecting the highest scoring.')
    agent.add_argument('-tr', '--truncate', type='bool', default=True,
                       help='truncate input & output lengths to speed up '
                       'training (may reduce accuracy). This fixes all '
                       'input and output to have a maximum length and to '
                       'be similar in length to one another by throwing '
                       'away extra tokens. This reduces the total amount '
                       'of padding in the batches.')
    agent.add_argument('-enc', '--encoder', default='gru',
                       choices=HredAgent.ENC_OPTS.keys(),
                       help='Choose between different encoder modules.')
    agent.add_argument('-bi', '--bi-encoder', default=True, type='bool',
                       help='Bidirection of encoder')
    agent.add_argument('-dec', '--decoder', default='same',
                       choices=['same', 'shared'] +
                       list(HredAgent.ENC_OPTS.keys()),
                       help='Choose between different decoder modules. '
                            'Default "same" uses same class as encoder, '
                            'while "shared" also uses the same weights.')
    agent.add_argument('-con', '--context', default='same',
                       choices=['same', 'shared'] +
                       list(HredAgent.ENC_OPTS.keys()),
                       help='Choose between different context modules. '
                       'Default "same" uses same class as encoder, '
                       'while "shared" also uses the same weights.')
    agent.add_argument('-opt', '--optimizer', default='sgd',
                       choices=Hred.OPTIM_OPTS.keys(),
                       help='Choose between pytorch optimizers. '
                            'Any member of torch.optim is valid and will '
                            'be used with default params except learning '
                            'rate (as specified by -lr).')
    agent.add_argument('-gradClip', '--grad-clip', type=float, default=-1,
                       help='gradient clip, default = -1 (no clipping)')
    agent.add_argument('-epi', '--episode-concat', type='bool', default=False,
                       help='If multiple observations are ' +
                       'from the same episode, concatenate them.')
    agent.add_argument('--beam_size', type=int, default=0,
                       help='Beam size for beam search ' +
                       '(only for generation mode) \n For Greedy search set 0')
    agent.add_argument('--max_seq_len', type=int, default=50,
                       help='The maximum sequence length, default = 50')
    agent.add_argument('-lt', '--lookuptable', default='all',
                       choices=['unique', 'enc_dec', 'dec_out', 'all'],
                       help='The encoder, decoder, and output modules can '
                            'share weights, or not. '
                            'Unique has independent embeddings for each. '
                            'Enc_dec shares the embedding for the encoder '
                            'and decoder. '
                            'Dec_out shares decoder embedding and output '
                            'weights. '
                            'All shares all three weights.')
    agent.add_argument('--embed', type=str, default=None,
                       help='pretrained embedding')
    agent.add_argument('--psize', type=int, default=2048,
                       help='projection size before the classifier')
    agent.add_argument('--pretrained_model_file', type=str, default=None,
                       help='pretrained model file for' +
                       ' the encoder and the decoder')
    agent.add_argument('--num_mechanisms', type=int, default=4,
                       help='The number of mechanisms')
    agent.add_argument('--mechanism_size', type=int, default=128,
                       help='projection size before the classifier')
    agent.add_argument('--nhead', type=int, default=8,
                       help='nhead of the transformer')

  def __init__(self, opt, shared=None):
    print('{}: 대화 모델 로딩 시작'.format(datetime.now()))
    """Set up model if shared params not set, otherwise no work to do."""
    super().__init__(opt, shared)
    if not shared:
      # this is not a shared instance of this class, so do full
      # initialization. if shared is set, only set up shared members.

      # check for cuda
      self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
      if self.use_cuda:
        print('[ Using CUDA ]')
        try:
          torch.cuda.set_device(int(opt['gpu']))
        except ValueError:
          pass

      self.states = {}
      if opt.get('model_file') and os.path.isfile(opt['model_file']):
        # load model parameters if available
        print('Loading existing model params from ' + opt['model_file'])
        new_opt, self.states = self.load(opt['model_file'])
        # override options with stored ones
        opt = self.override_opt(new_opt)

      if opt.get('dict_class'):
        self.dict = str2class(opt['dict_class'])(opt)
      else:
        self.dict = DictionaryAgent(opt)
      self.id = 'Hred'
      # we use START markers to start our output
      self.START = self.dict.start_token
      self.START_IDX = self.dict.parse(self.START)
      # we use END markers to end our output
      self.END = self.dict.end_token
      # get index of null token from dictionary (probably 0)
      self.NULL_IDX = self.dict.txt2vec(self.dict.null_token)[0]

      opt['dict'] = self.dict

      self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))
      # we use END markers to end our output
      self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))

      # store important params directly
      hsz = opt['hiddensize']
      emb = opt['embeddingsize']
      self.hidden_size = hsz
      self.emb_size = emb
      self.num_layers = opt['numlayers']
      self.learning_rate = opt['learning_rate']
      self.longest_label = self.states.get('longest_label', 1)
      self.truncate = opt['truncate']
      self.attention = opt['attention']

      # set up tensors
      self.xses = []
      self.ys = torch.LongTensor(1, 1)

      print('NLLLoss 인스턴스 생성 시작.')
      self.criterion = nn.NLLLoss(size_average=False, ignore_index=0)
      print('NLLLoss 인스턴스 생성 완료.')

      # set up modules
      self.model = Hred(opt, len(self.dict), self.START_IDX,
                        self.NULL_IDX, longest_label=self.longest_label)

      if opt.get('pretrained_model_file'):
        print('Loading pretrained model params from ' +
              opt['pretrained_model_file'])
        _, pretrained_states = self.load(opt['pretrained_model_file'])
        pretrained_model = Seq2seq(
            opt, len(self.dict), self.START_IDX,
            self.NULL_IDX, longest_label=self.longest_label)
        pretrained_model.load_state_dict(pretrained_states['model'])
        self.model.lt.load_state_dict(pretrained_model.lt.state_dict())
        self.model.encoder.load_state_dict(
            pretrained_model.encoder.state_dict())
        self.model.decoder.load_state_dict(
            pretrained_model.decoder.state_dict())
        if type(self.model.o2e) is nn.Linear:
          self.model.o2e.load_state_dict(pretrained_model.o2e.state_dict())
        self.model.e2s.load_state_dict(pretrained_model.e2s.state_dict())

      self.zeros_dec = torch.zeros(self.num_layers, 1, hsz)
      self.use_attention = False

      # initialization
      """
                getattr(self, 'lt').weight.data.uniform_(-0.1, 0.1)
                for module in {'encoder', 'decoder'}:
                    for weight in getattr(self, module).parameters():
                        weight.data.normal_(0, 0.05)
                    # for bias in getattr(self, module).parameters():
                    #    bias.data.fill_(0)

                for module in {'h2o', 'attn'}:
                    if hasattr(self, module):
                        getattr(self, module).weight.data.normal_(0, 0.01)
                        # getattr(self, module).bias.data.fill_(0)
            """

      # set up optims for each module
      self.wd = opt['weight_decay']

      if self.states:
        # set loaded states if applicable
        self.set_states(self.states)

      if self.use_cuda:
        self.cuda()

      self.loss = 0
      self.loss_valid = 0
      self.ndata = 0
      self.ndata_valid = 0

      kwargs = {'lr': opt['learning_rate']}
      if opt['optimizer'] == 'sgd':
        kwargs['momentum'] = 0.95
        kwargs['nesterov'] = True

      optim_class = Hred.OPTIM_OPTS[opt['optimizer']]
      self.optimizer = optim_class(self.model.parameters(), **kwargs)

      if self.states:
        if self.states['optimizer_type'] != opt['optimizer']:
          print('WARNING: not loading optim state since optim class '
                'changed.')
        else:
          self.optimizer.load_state_dict(self.states['optimizer'])

      if opt['beam_size'] > 0:
        self.beamsize = opt['beam_size']

    self.episode_concat = opt['episode_concat']
    self.training = True
    self.generating = False
    self.local_human = False

    if opt.get('max_seq_len') is not None:
      self.max_seq_len = opt['max_seq_len']
    else:
      self.max_seq_len = opt['max_seq_len'] = 50
    self.reset()
    print('{}: 대화 모델 로딩 완료'.format(datetime.now()))

    try:
      task_module = importlib.import_module(
          'parlai.tasks.{}.build'.format(self.opt['task']))
      self.preprocess = task_module.preprocess
      if self.opt['task'] == 'opensubtitles_ko_nlg' and self.local_human:
        self.syllable = self.preprocess
        self.morphs = task_module.morphs
        self.preprocess = lambda x: self.syllable(self.morphs(x))
      try:
        self.postprocess = importlib.import_module(
            'parlai.tasks.{}.build'.format(self.opt['task'])).postprocess
      except AttributeError:
        self.postprocess = lambda x: x
    except ModuleNotFoundError:
      self.preprocess = lambda x: x
      self.postprocess = lambda x: x

  def override_opt(self, new_opt):
    """Set overridable opts from loaded opt file.

    Print out each added key and each overriden key.
    Only override args specific to the model.
    """
    model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                  'encoder', 'decoder'}
    for k, v in new_opt.items():
      if k not in model_args:
        # skip non-model args
        continue
      if k not in self.opt:
        print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
      elif self.opt[k] != v:
        print('Overriding option [ {k}: {old} => {v}]'.format(
              k=k, old=self.opt[k], v=v))
      self.opt[k] = v
    return self.opt

  def parse(self, text):
    """Convert string to token indices."""
    return self.dict.txt2vec(text)

  def v2t(self, vec):
    """Convert token indices to string of tokens."""
    return self.dict.vec2txt(vec)

  def zero_grad(self):
    self.optimizer.zero_grad()

  def cuda(self):
    """Push parameters to the GPU."""
    self.model.cuda()
    self.xses = [xs.cuda(non_blocking=True) for xs in self.xses]
    if type(self.opt['gpu']) is str and ',' in self.opt['gpu']:
      last_index = int(self.opt['gpu'].split(',')[-1])
      self.criterion.cuda(last_index)
      self.ys = self.ys.cuda(last_index, non_blocking=True)
    else:
      self.ys = self.ys.cuda(non_blocking=True)
      self.criterion.cuda()
    self.START_TENSOR = self.START_TENSOR.cuda(non_blocking=True)
    self.END_TENSOR = self.END_TENSOR.cuda(non_blocking=True)
    self.START_TENSOR = self.START_TENSOR.cuda(non_blocking=True)
    self.END_TENSOR = self.END_TENSOR.cuda(non_blocking=True)

  def reset(self):
    """Reset observation and episode_done."""
    self.observation = None
    self.episode_done = True

  def observe(self, observation):
    """Save observation for act.
    If multiple observations are from the same episode, concatenate them.
    """
    if self.local_human:
      observation = {}
      observation['id'] = self.getID()
      try:
        reply_text = input("Enter Your Message: ")
      except UnicodeDecodeError:
        reply_text = '디코딩 에러가 났습니다.'
      reply_text = self.preprocess(reply_text)
      observation['episode_done'] = True  # ## TODO: for history

      """
            if '[DONE]' in reply_text:
                reply['episode_done'] = True
                self.episodeDone = True
                reply_text = reply_text.replace('[DONE]', '')
            """
      observation['text'] = reply_text
    else:
      # shallow copy observation (deep copy can be expensive)
      observation = observation.copy()
      if not self.episode_done and self.episode_concat:
        # if the last example wasn't the end of an episode, then we need to
        # recall what was said in that example
        prev_dialogue = self.observation['text']
        observation['text'] = prev_dialogue + '\n' + \
            observation['text']  # TODO!!!! # DATA is concatenated!!

    self.observation = observation
    self.episode_done = observation['episode_done']

    return observation

  def batch_observe(self, observations):
    self.observations = observations.copy()

  def update_params(self):
    """Do one optimization step."""
    self.optimizer.step()

  def predict(self, xses, xlens, ylen=None, ys=None, m_idx=None):
    """Produce a prediction from our model.

    Update the model using the targets if available, otherwise rank
    candidates as well if they are available.
    """

    self.model.train(self.training)
    self.zero_grad()

    # from .mem_report import mem_report
    # mem_report()

    batchsize = len(xses[0])

    xlen_ts = [torch.LongTensor(xlen) - 1 for xlen in xlens]
    if self.use_cuda:
      xlen_ts = [xlen_t.cuda() for xlen_t in xlen_ts]
    xlen_ts = [Variable(xlen_t, requires_grad=False) for xlen_t in xlen_ts]

    beam_cands = None

    if self.training:
      scores, preds = self.model(xses, self.training, xlen_ts, ys)

      if ys is not None:
        loss = 0
        for i, score in enumerate(scores):
          y = ys.select(1, i)
          loss += self.criterion(score, y)

        self.loss = loss.item() / sum(ylen)
        self.ndata += batchsize

      output_lines = [[] for _ in range(batchsize)]
      for b in range(batchsize):
        # convert the output scores to tokens
        output_lines[b] = self.v2t(preds.data[:, b])

      loss.backward()

      if self.opt['grad_clip'] > 0:
        torch.nn.utils.clip_grad_norm(
            self.model.lt.parameters(), self.opt['grad_clip'])
        torch.nn.utils.clip_grad_norm(
            self.model.h2o.parameters(), self.opt['grad_clip'])
        torch.nn.utils.clip_grad_norm(
            self.model.encoder.parameters(), self.opt['grad_clip'])
        torch.nn.utils.clip_grad_norm(
            self.model.decoder.parameters(), self.opt['grad_clip'])
      self.update_params()
# HEM Start
      self.display_predict(xses, ys, output_lines, 0)
    else:
      with torch.no_grad():
        if self.opt['beam_size'] > 0:
          context_hidden = None
          for idx in range(0, len(xses)):
            hidden = self.model._encode(xses[idx], xlen_ts[idx], self.training)
            mask = Hred.to_mask(xses[idx])
            context_hidden = self.model._context(
                context_hidden, hidden, mask)

          transposed_context_hidden = context_hidden.transpose(0, 1)

          if m_idx is None:
            alphas = self.model.softmax(
                self.model.merge(transposed_context_hidden))
            merged = (alphas.transpose(1, 2) @
                      transposed_context_hidden).squeeze(1)

            t = self.model.max_out(merged)
            w = self.model.w_t(t)
            g = w @ self.model.mechanisms
            p_m = self.model.softmax(g)
            m = p_m @ self.model.mechanisms.t()
          else:
            m = self.model.mechanisms[:, m_idx, None].t()

          m = self.model.to_emb(m).unsqueeze(0)

          hidden = torch.cat((hidden, context_hidden, m), dim=0)
          mask = torch.cat((mask, torch.BoolTensor(batchsize, 2).new_full(
              (batchsize, 2), False
          ).cuda(mask.get_device())), dim=1)

          x = Variable(self.model.START, requires_grad=False)
          xe = self.model.lt(x).unsqueeze(1)
          dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

          output_lines, beam_cands = self._beam_search(
              batchsize, dec_xes, hidden, mask)
        else:
          beam_cands = []
          scores, preds = self.model(xses, self.training, xlen_ts, ys)

          if ys is not None:
            loss = 0
            losses = []
            for i, score in enumerate(scores):
              y = ys.select(1, i)
              print('NLLLoss API 사용 시작')
              losses.append(self.criterion(score, y))
              loss += losses[i]
              print('NLLLoss API 사용 완료')

              self.loss_valid += loss.item()
              self.ndata_valid += sum(ylen)

          output_lines = [[] for _ in range(batchsize)]
          for b in range(batchsize):
            # convert the output scores to tokens
            output_lines[b] = self.v2t(preds.data[:, b])
          self.display_predict(xses, ys, None, 1, losses, ylen)

    return output_lines, beam_cands

  def display_predict(self, xses, ys, output_lines, freq=0.01, losses=None,
                      ylen=None):
    if random.random() < freq:
      # sometimes output a prediction for debugging
      for idx in range(len(ys)):
        inputs = []
        for xs in xses:
          x = self.dict.vec2txt(xs[idx].data.cpu()).replace(
              self.dict.null_token + ' ', '')
          x = re.sub('__START__ ', '', x)
          x = re.sub(' __END__.*', '', x)
          x = re.sub('__NULL__', '', x)
          if x:
            inputs.append(x)
        if output_lines:
          p = output_lines[idx]
        else:
          p = None
        l = self.dict.vec2txt(ys[idx].data.cpu()).replace(
            self.dict.null_token + ' ', '')
        l = re.sub('__START__ ', '', l)
        l = re.sub(' __END__.*', '', l)
        l = re.sub('__NULL__', '', l)
        print('\n    input:', ' __SEP__ '.join(inputs))
        if ys is not None:
          print('    label:', l, '\n')
        if losses is not None:
          print('    NLLLoss: {}'.format(losses[idx].item()))
        if ylen is not None:
          print('    ylen: {}'.format(ylen[idx]))
        print('{}\t{}\t{}\t{}'.format(
            ' __SEP__ '.join(inputs), l, sum(losses).item(), ylen[idx]))

  def batchify(self, observations):
    """Convert a list of observations into input & target tensors."""
    # valid examples
    exs = [ex for ex in observations if 'text' in ex]
    # the indices of the valid (non-empty) tensors
    valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

    # set up the input tensors
    batchsize = len(exs)

    # tokenize the text
    if batchsize > 0:
      parsed = [[self.START_IDX +
                 self.parse(text) +
                 self.dict.parse(self.END) for text in ex['text'].split('\n')]
                for ex in exs]
      max_xes_len = max([len(xes) for xes in parsed])
      for idx, xes in enumerate(parsed):
        if len(xes) < max_xes_len:
          parsed[idx] = [[]] * (max_xes_len - len(xes)) + xes
      max_x_len = max([len(x) for x in xes for xes in parsed])
      if self.truncate:
        # shrink xs to to limit batch computation
        max_x_len = min(max_x_len, self.max_seq_len)
        parsed = [[x[-max_x_len:] for x in xes] for xes in parsed]

      parsed_xes = [parsed for parsed in zip(*parsed)]
      xlens = [[len(x) for x in parsed_x] for parsed_x in parsed_xes]
      xses = [torch.LongTensor(batchsize, max_x_len).fill_(0) for _ in xlens]

      for k, parsed_x in enumerate(parsed_xes):
        for i, x in enumerate(parsed_x):
          for j, idx in enumerate(x):
            xses[k][i][j] = idx
          if len(x) == 0:
            xses[k][i][0] = self.START_IDX[0]
            xses[k][i][1] = self.dict.parse(self.END)[0]

      if self.use_cuda:
        # copy to gpu
        while len(self.xses) < len(xses):
          self.xses.append(torch.LongTensor(1, 1).cuda(non_blocking=True))

        for idx, xs in enumerate(xses):
          self.xses[idx].resize_(xs.size())
          self.xses[idx].copy_(xs, non_blocking=True)
        xses = [Variable(self.xses[idx], requires_grad=False)
                for idx, _ in enumerate(xses)]
      else:
        xses = [Variable(xs, requires_grad=False) for xs in xses]
    else:
      import pdb
      pdb.set_trace()

    # set up the target tensors
    ys = None
    ylen = None

    if batchsize > 0 and not self.generating:
      # randomly select one of the labels to update on, if multiple
      # append END to each label
      if any(['labels' in ex for ex in exs]):
        labels = [random.choice(ex.get('labels', [''])) +
                  ' ' + self.END for ex in exs]
      else:
        labels = [random.choice(
            ex.get('eval_labels', [''])) + ' ' + self.END for ex in exs]

      parsed_y = [self.parse(y) for y in labels]
      max_y_len = max(len(y) for y in parsed_y)
      if self.truncate:
        # shrink ys to to limit batch computation
        max_y_len = min(max_y_len, self.max_seq_len)
        parsed_y = [y[:max_y_len] for y in parsed_y]

      ylen = [len(x) for x in parsed_y]
      ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
      for i, y in enumerate(parsed_y):
        for j, idx in enumerate(y):
          ys[i][j] = idx
      if self.use_cuda:
        # copy to gpu
        self.ys.resize_(ys.size())
        self.ys.copy_(ys, non_blocking=True)
        ys = Variable(self.ys, requires_grad=False)
      else:
        ys = Variable(ys, requires_grad=False)
    return xses, ys, valid_inds, xlens, ylen

  def batch_act(self, observations):
    batchsize = len(observations)
    # initialize a table of replies with this agent's id
    batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

    # convert the observations into batches of inputs and targets
    # valid_inds tells us the indices of all valid examples
    # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
    # since the other three elements had no 'text' field
    # xs, ys, valid_inds, xlen, ylen = self.batchify(observations)
    xses, ys, valid_inds, xlens, ylen = self.batchify(observations)

    if xses is None:
      # no valid examples, just return the empty responses we set up
      return batch_reply

    # produce predictions either way, but use the targets if available
    self.training = any('labels' in obs for obs in observations)
    predictions, beam_cands = self.predict(xses, xlens, ylen, ys)

    if self.local_human:
      print(self.postprocess(predictions[0]))

    for i in range(len(predictions)):
      # map the predictions back to non-empty examples in the batch
      # we join with spaces since we produce tokens one at a time
      curr = batch_reply[valid_inds[i]]
      curr['text'] = ''.join(c for c in predictions[i] if c !=
                             self.END and c != self.dict.null_token)

    return batch_reply

  def batch_beam_act(self, observations):
    batchsize = len(observations)
    # initialize a table of replies with this agent's id
    batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

    # convert the observations into batches of inputs and targets
    # valid_inds tells us the indices of all valid examples
    # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
    # since the other three elements had no 'text' field
#        xs, ys, valid_inds, xlen, ylen = self.batchify(observations)
    xses, ys, valid_inds, xlens, ylen = self.batchify(observations)

    if xses is None:
      # no valid examples, just return the empty responses we set up
      return batch_reply

    # produce predictions either way, but use the targets if available

    # mechanism
    if batchsize and 'mechanism' in observations[0]:
      m_idx = observations[0]['mechanism']
      predictions, beam_cands = self.predict(xses, xlens, ylen, ys, m_idx)
    else:
      predictions, beam_cands = self.predict(xses, xlens, ylen, ys)

    if self.local_human:
      print(self.postprocess(predictions[0]))

    return predictions

  def act(self):
    # call batch_act with this batch of one
    return self.batch_act([self.observation])[0]

  def act_beam_cands(self):
    return self.batch_beam_act([self.observation])

  def batch_act_beam_cands(self):
    return self.batch_beam_act(self.observations)

  def save(self, path=None):
    path = self.opt.get('model_file', None) if path is None else path

    if path and hasattr(self, 'model'):
      model = {}
      model['model'] = self.model.state_dict()
      model['longest_label'] = self.model.longest_label
      model['opt'] = self.opt
      model['optimizer'] = self.optimizer.state_dict()
      model['optimizer_type'] = self.opt['optimizer']

      os.makedirs(os.path.dirname(path), exist_ok=True)

      with open(path, 'wb') as write:
        torch.save(model, write)

  def shutdown(self):
    """Save the state of the model when shutdown."""
    path = self.opt.get('model_file', None)
    if path is not None:
      self.save(path + '.shutdown_state')
    super().shutdown()

  def load(self, path):
    """Return opt and model states."""
    with open(path, 'rb') as read:
      model = torch.load(read, map_location={'cuda:1': 'cuda:0'})
    return model['opt'], model

  def set_states(self, states):
    """Set the state dicts of the modules from saved states."""
    self.model.load_state_dict(states['model'])

  def report(self):
    m = {}
    if not self.generating:
      if self.training:
        m['nll'] = self.loss
        m['ppl'] = math.exp(self.loss)
        m['ndata'] = self.ndata
      else:
        m['nll'] = self.loss_valid / self.ndata_valid
        m['ppl'] = math.exp(self.loss_valid / self.ndata_valid)
        print(('계산된 NLLLoss({})에 총 출력 단어수({})를 나눠준 후' +
               ' exponential을 취하여 perplexity 계산.').format(
            self.loss_valid, self.ndata_valid))
        m['ndata'] = self.ndata_valid

      m['lr'] = self.optimizer.param_groups[0]['lr']
      # self.print_weight_state()

    return m

  def reset_valid_report(self):
    self.ndata_valid = 0
    self.loss_valid = 0

  def print_weight_state(self):
    self._print_grad_weight(getattr(self, 'lt').weight, 'lookup')
    for module in {'encoder', 'decoder'}:
      layer = getattr(self, module)
      for weights in layer._all_weights:
        for weight_name in weights:
          self._print_grad_weight(
              getattr(layer, weight_name), module + ' ' + weight_name)
    self._print_grad_weight(getattr(self, 'h2o').weight, 'h2o')
    if self.use_attention:
      self._print_grad_weight(getattr(self, 'attn').weight, 'attn')

  def _print_grad_weight(self, weight, module_name):
    if weight.dim() == 2:
      nparam = weight.size(0) * weight.size(1)
      norm_w = weight.norm(2).pow(2)
      norm_dw = weight.grad.norm(2).pow(2)
      print('{:30}'.format(module_name) + ' {:5} x{:5}'.format(weight.size(0),
                                                               weight.size(1))
            + ' : w {0:.2e} | '.format((norm_w / nparam).sqrt().data[0]) +
            'dw {0:.2e}'.format((norm_dw / nparam).sqrt().data[0]))

  def _get_context(self, batchsize, xlen_t, encoder_output):
    " return initial hidden of decoder and encoder context (last_state)"

    if self.zeros_dec.size(1) != batchsize:
      self.zeros_dec.resize_(self.model.num_layers,
                             batchsize, self.model.hidden_size).fill_(0)
    hidden = Variable(self.zeros_dec.fill_(0), requires_grad=False)

    last_state = None
    if not self.use_attention:
      last_state = torch.gather(
          encoder_output,
          1,
          xlen_t.view(-1, 1, 1).expand(encoder_output.size(0),
                                       1,
                                       encoder_output.size(2)))
      if self.opt['bi_encoder']:
        last_state = torch.cat(
            (encoder_output[:, 0, self.hidden_size:],
             last_state[:, 0, :self.hidden_size]), 1)

    return hidden, last_state

  def _beam_search(self, batchsize, dec_xes,
                   hidden, all_hidden_mask, n_best=20):
    # Code borrowed from PyTorch OpenNMT example`
    # https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/decode.py

    # just produce a prediction without training the model
    done = [False for _ in range(batchsize)]
    total_done = 0
    max_len = 0
    output_lines = [[] for _ in range(batchsize)]

    # exapnd tensors for each beam
    beamsize = self.beamsize

    dec_states = [
        Variable(hidden.data.repeat(1, beamsize, 1))  # seq, beamsize, emb
    ]

    beam = [Beam(beamsize, self.dict.tok2ind, cuda=self.use_cuda)
            for k in range(batchsize)]

    remaining_sents = batchsize

    input = Variable(dec_xes.data.repeat(1, beamsize, 1))
    # encoder_output = Variable(encoder_output.data.repeat(beamsize, 1, 1))

    decoder_gpu = next(self.model.decoder.parameters()).get_device()

    ys = self.model.START.expand(1, beamsize * batchsize)

    all_hidden_mask = all_hidden_mask.repeat(beamsize, 1)
    if all_hidden_mask.get_device() != decoder_gpu:
      all_hidden_mask = all_hidden_mask.cuda(decoder_gpu)

    while total_done < batchsize and max_len < self.model.longest_label:
      not_done_beams = [b for b in beam if not b.done]
      remaining_sents = len(not_done_beams)

      h = torch.cat([dec_states[0][:, i * beamsize:(i + 1) * beamsize, :]
                     for i, b in enumerate(beam)if not b.done], 1)
      hidden_mask = torch.cat([
          all_hidden_mask[i * beamsize:(i + 1) * beamsize, :]
          for i, b in enumerate(beam) if not b.done])
      if decoder_gpu != input.get_device():
        input = input.cuda(decoder_gpu)
      output = self.model.decoder(
          input, h,
          tgt_mask=self.model.generate_square_subsequent_mask(
              input.size(0)).cuda(decoder_gpu),
          tgt_key_padding_mask=Hred.to_mask(
              ys.transpose(0, 1)).cuda(decoder_gpu),
          memory_key_padding_mask=hidden_mask)
      preds, scores = self.model.hiddens_to_idx(
          output.index_select(
              0, torch.tensor([output.size(0) - 1]).cuda(decoder_gpu)),
          dropout=False)

      # dec_states = [hidden]
      word_lk = scores.view(beamsize, remaining_sents, -
                            1).transpose(0, 1).contiguous()  # 1, beamsize, -1

      active = False
      for i, b in enumerate(not_done_beams):
        if not b.advance(word_lk.data[i]):
          active = True

      if not active:
        break

      ys = torch.cat(
          [ys.index_select(
              1, b.get_current_origin() + i * beamsize
          ) for i, b in enumerate(not_done_beams) if not b.done], 1)
      ys = torch.cat(
          (ys,
           torch.stack(
               [b.get_current_state()
                for b in not_done_beams
                if not b.done]).t().contiguous().view(1, -1)))
      input = self.model.lt(Variable(ys))
      input = self.model.pos_encoder(input)

      max_len += 1

    all_preds, allScores = [], []
    for b in range(batchsize):  # TODO :: does it provide batchsize > 1 ?
      hyps = []
      scores, ks = beam[b].sort_best()

      allScores += [scores[:self.beamsize]]
      hyps += [beam[b].get_hyp(k) for k in ks[:self.beamsize]]

      all_preds += [' '.join([self.dict.ind2tok[y.item()] for y in x if not y is 0])
                    for x in hyps]

    return all_preds[0::beamsize], all_preds  # 1-best
