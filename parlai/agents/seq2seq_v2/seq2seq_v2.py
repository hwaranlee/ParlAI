# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.tasks.acryl_korean.build import preprocess
# from .beam import Beam
from .beam_diverse import Beam
from .modules import Seq2seq

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch

import os
import random, math
import pdb


class Seq2seqV2Agent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    For more information, see Sequence to Sequence Learning with Neural
    Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
    """

    ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
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
                           help='general=bilinear dotproduct, concat=bahdanau\'s implemenation')        
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
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
                           choices=Seq2seqV2Agent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-bi', '--bi-encoder', default=True, type='bool',
                           help='Bidirection of encoder')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(Seq2seqV2Agent.ENC_OPTS.keys()),
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=Seq2seq.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-gradClip', '--grad-clip', type=float, default=-1,
                       help='gradient clip, default = -1 (no clipping)')
        agent.add_argument('-epi', '--episode-concat', type='bool', default=False,
                       help='If multiple observations are from the same episode, concatenate them.')
        agent.add_argument('--beam_size', type=int, default=0,
                           help='Beam size for beam search (only for generation mode) \n For Greedy search set 0')
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
        agent.add_argument('--embed', type=str, default=None, help='pretrained embedding')
                
    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            self.states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)
                
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START = self.dict.start_token
            self.START_IDX = self.dict.parse(self.START)
            # we use END markers to end our output
            self.END = self.dict.end_token
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict.txt2vec(self.dict.null_token)[0]

            opt['dict'] = self.dict

            # store important params directly
            hsz = opt['hiddensize']
            emb = opt['embeddingsize']
            self.hidden_size = hsz
            self.emb_size = emb
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learning_rate']
            self.rank = opt['rank_candidates']
            self.longest_label = self.states.get('longest_label', 1)
            self.truncate = opt['truncate']
            self.attention = opt['attention']

            # set up tensors
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)

            self.criterion = nn.NLLLoss(size_average=False, ignore_index=0)

            # set up modules
            self.model = Seq2seq(opt, len(self.dict), self.START_IDX, self.NULL_IDX, longest_label=self.longest_label)
            # self.model = nn.DataParallel(Seq2seq(opt, len(self.dict), self.START_IDX, self.NULL_IDX, longest_label=self.states.get('longest_label', 1)), [0])
            
            self.use_attention = False

            # initialization
            """
                getattr(self, 'lt').weight.data.uniform_(-0.1, 0.1)
                for module in {'encoder', 'decoder'}:
                    for weight in getattr(self, module).parameters():
                        weight.data.normal_(0, 0.05)
                    #for bias in getattr(self, module).parameters():
                    #    bias.data.fill_(0)
                        
                for module in {'h2o', 'attn'}:
                    if hasattr(self, module):
                        getattr(self, module).weight.data.normal_(0, 0.01)
                        #getattr(self, module).bias.data.fill_(0)
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
            
            optim_class = Seq2seq.OPTIM_OPTS[opt['optimizer']]
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
        self.xs = self.xs.cuda(async=True)
        if self.opt['split_gpus']:
            self.criterion.cuda(1)
            self.ys = self.ys.cuda(1, async=True)
        else:
            self.ys = self.ys.cuda(async=True)
            self.criterion.cuda()

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
            reply_text = input("Enter Your Message: ")
            reply_text = preprocess(reply_text)
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
                observation['text'] = prev_dialogue + '\n' + observation['text']  #### TODO!!!! # DATA is concatenated!!
  
        self.observation = observation
        self.episode_done = observation['episode_done']          
        
        return observation

    def update_params(self):
        """Do one optimization step."""
        self.optimizer.step()

    def predict(self, xs, xlen, ylen=None, ys=None, cands=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        
        self.model.train(self.training)
        self.zero_grad()
        
        batchsize = len(xs)
        text_cand_inds = None
        
        xlen_t = torch.LongTensor(xlen) - 1
        if self.use_cuda:
            xlen_t = xlen_t.cuda()
        xlen_t = Variable(xlen_t)

        scores, preds = self.model(xs, self.training, xlen_t, ys)

        if ys is not None:
            loss = 0
            for i, score in enumerate(scores):
                y = ys.select(1, i)
                loss += self.criterion(score, y)
    
            if self.training:
                self.loss = loss.data[0] / sum(ylen)
                self.ndata += batchsize
            else:
                self.loss_valid += loss.data[0]
                self.ndata_valid += sum(ylen)
            
        output_lines = [[] for _ in range(batchsize)]
        for b in range(batchsize):
            # convert the output scores to tokens
            output_lines[b] = self.v2t(preds.data[b])
        
        if self.training:
            loss.backward()
            if self.opt['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm(self.model.lt.parameters(), self.opt['grad_clip'])
                torch.nn.utils.clip_grad_norm(self.model.h2o.parameters(), self.opt['grad_clip'])                    
                torch.nn.utils.clip_grad_norm(self.model.encoder.parameters(), self.opt['grad_clip'])
                torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), self.opt['grad_clip'])
            self.update_params()

#        try:
#            print(output_lines[0].split().index('__END__'))
#        except ValueError:
#            print(len(output_lines[0].split()))
        
        self.display_predict(xs, ys, output_lines, 0)

        if ys is None:
            print(output_lines)

        return output_lines, text_cand_inds

    def display_predict(self, xs, ys, output_lines, freq=0.01):
        if random.random() < freq:
            # sometimes output a prediction for debugging
            print('\n    input:', self.dict.vec2txt(xs[0].data.cpu()).replace(self.dict.null_token + ' ', ''),
                  '\n    pred :', ' '.join(output_lines[0]), '\n')
            if ys is not None:
                print('    label:', self.dict.vec2txt(ys[0].data.cpu()).replace(self.dict.null_token + ' ', ''), '\n')

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        xs = None
        xlen = None
        if batchsize > 0:
            parsed = [self.START_IDX + self.parse(ex['text']) + self.dict.parse(self.END) for ex in exs]
            max_x_len = max([len(x) for x in parsed])            
            if self.truncate:
                # shrink xs to to limit batch computation
                max_x_len = min(max_x_len, self.max_seq_len)
                parsed = [x[-max_x_len:] for x in parsed]
        
            # sorting for unpack in encoder
            parsed_x = sorted(parsed, key=lambda p: len(p), reverse=True)            
            xlen = [len(x) for x in parsed_x]            
            xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
            
            """
            # pack the data to the right side of the tensor for this model
            for i, x in enumerate(parsed):
                offset = max_x_len - len(x)
                for j, idx in enumerate(x):
                    xs[i][j + offset] = idx
                    """
            for i, x in enumerate(parsed_x):
                for j, idx in enumerate(x):
                    xs[i][j] = idx        
            if self.use_cuda:
                # copy to gpu
                self.xs.resize_(xs.size())
                self.xs.copy_(xs, async=True)
                xs = Variable(self.xs)
            else:
                xs = Variable(xs)
            
        # set up the target tensors
        ys = None
        ylen = None
        
        if batchsize > 0 and not self.generating:
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            if any(['labels' in ex for ex in exs]):
                labels = [random.choice(ex.get('labels', [''])) + ' ' + self.END for ex in exs]
            else:
                labels = [random.choice(ex.get('eval_labels', [''])) + ' ' + self.END for ex in exs]
                
            parsed_y = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed_y)
            if self.truncate:
                # shrink ys to to limit batch computation
                max_y_len = min(max_y_len, self.max_seq_len)
                parsed_y = [y[:max_y_len] for y in parsed_y]
            
            seq_pairs = sorted(zip(parsed, parsed_y), key=lambda p: len(p[0]), reverse=True)
            _, parsed_y = zip(*seq_pairs)
                            
            ylen = [len(x) for x in parsed_y]
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed_y):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, async=True)
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)
                
        return xs, ys, valid_inds, xlen, ylen

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds, xlen, ylen = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        
        predictions, text_cand_inds = self.predict(xs, xlen, ylen, ys)
        
        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            # curr['text'] = ' '.join(c for c in predictions[i] if c != self.END and c != self.dict.null_token) ## TODO: check!!
            curr['text'] = ''.join(c for c in predictions[i] if c != self.END and c != self.dict.null_token)  # # TODO: check!!

        if text_cand_inds is not None:
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]
        
        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def act_beam_cands(self):
        return self.batch_act([self.observation])[1]
    
    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['opt'] = self.opt
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']

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
            if(self.use_cuda):
                model = torch.load(read)
            else:
                model = torch.load(read, map_location=lambda storage, loc: storage)
        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.model.load_state_dict(self.states['model'])

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
                    self._print_grad_weight(getattr(layer, weight_name), module + ' ' + weight_name)
        self._print_grad_weight(getattr(self, 'h2o').weight, 'h2o')
        if self.use_attention:
           self._print_grad_weight(getattr(self, 'attn').weight, 'attn')
                
    def _print_grad_weight(self, weight, module_name):
        if weight.dim() == 2:
            nparam = weight.size(0) * weight.size(1)
            norm_w = weight.norm(2).pow(2)
            norm_dw = weight.grad.norm(2).pow(2)
            print('{:30}'.format(module_name) + ' {:5} x{:5}'.format(weight.size(0), weight.size(1))
                   + ' : w {0:.2e} | '.format((norm_w / nparam).sqrt().data[0]) + 'dw {0:.2e}'.format((norm_dw / nparam).sqrt().data[0]))
    
