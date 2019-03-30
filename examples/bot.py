import torch
import sys

from parlai.core.agents import create_agent
from parlai.core.worlds import validate
from parlai.core.params import ParlaiParser

import random
import logging
import sys
import os
import re


class Bot:
  def __init__(self, model_path, dict_dir, cuda=False):
    opt = get_opt(model_path, cuda)
    opt['model_file'] = model_path
    opt['datatype'] = 'valid'
    opt['dict_file'] = dict_dir
    opt['gpu'] = 0 if cuda else -1  # use cpu
    opt['cuda'] = cuda
    opt['no_cuda'] = (not cuda)
    opt['batchsize'] = 1
    opt['dict_class'] = 'parlai.tasks.ko_multi.dict:Dictionary'
    opt['beam_size'] = 7

    self.opt = opt
    self.agent = create_agent(opt)
    self.agent.training = False
    self.agent.generating = True

    self.user_history = {}
    self.histories = {}

  def reply(self, message, *args):
    if isinstance(message, list):
      observation = {}
      message = [self.agent.preprocess(m) for m in message]
      observation['episode_done'] = True
      emotion = ['surprised' if e == 'Surprise' else e for e in args[0]]
      observation['text'] = '\n'.join(
          [m + ' ' + e for m, e in zip(message, emotion)])
      self.agent.observe(validate(observation))
      response = self.agent.act_beam_cands()
      response = self.agent.postprocess(response[0])
      splited = response.split()
      emotion = splited[-1]
      if emotion == 'surprised':
        emotion = 'Surprise'
      if emotion in ('Neutral', 'Surprise', 'Anger', 'Sadness', 'Fear',
                     'Happiness', 'Disgust'):
        response = ' '.join(splited[:-1])
      else:
        emotion = 'Neutral'

      return response, emotion
    else:
      observation = {}
      message = self.agent.preprocess(message)
      observation['episode_done'] = True  # TODO: for history
      if len(args) > 0:
        if args[0] == 'Surprise':
          emotion = 'surprised'
        else:
          emotion = args[0]

        if len(args) > 1:
          id = args[1]
          if not id in self.histories:
            self.histories[id] = []
          self.histories[id].append(message + ' ' + emotion)
          self.histories[id] = self.histories[id][-5:]
          observation['text'] = '\n'.join(self.histories[id])
        else:
          observation['text'] = message + ' ' + emotion

        self.agent.observe(validate(observation))
        response = self.agent.act_beam_cands()

        if(message in self.user_history):
          idx = self.user_history[message] % 7
        else:
          idx = 0

        if len(args) > 1:
          self.histories[id].append(re.sub(' __END__.*', '', response[idx]))

        response = self.agent.postprocess(response[idx])
        self.user_history[message] = idx + 1
      else:
        observation['text'] = message

        self.agent.observe(validate(observation))
        response = self.agent.act()
        response = self.agent.postprocess(response['text'])

      if len(args) > 0 and response != '':
        splited = response.split()
        emotion = splited[-1]
        if emotion == 'surprised':
          emotion = 'Surprise'
        if emotion in ('Neutral', 'Surprise', 'Anger', 'Sadness', 'Fear',
                       'Happiness', 'Disgust'):
          response = ' '.join(splited[:-1])
        else:
          emotion = 'Neutral'

        return response, emotion
      else:
        return response


def get_opt(model_path, cuda=False):
  if cuda:
    mdl = torch.load(model_path)
  else:
    mdl = torch.load(model_path, map_location=lambda storage, loc: storage)

  opt = mdl['opt']
  del mdl

  return opt


if __name__ == "__main__":

  root_dir = '../ParlAI-v2'
  model_name = 'exp-emb200-hs2048-lr0.0001-allK'
  pretrained_mdl_path = os.path.join(
      root_dir, 'exp/', model_name, model_name)  # release ver
  dict_dir = os.path.join(
      root_dir, 'exp-opensub_kemo_all/dict_file_100000.dict')

  cc = Bot(pretrained_mdl_path, dict_dir, cuda=True)

  print(cc.reply('지금 모하는 거니?', 'Neutral'))
  print(cc.reply('지금 모하는 거니?', 'surprised'))
  print(cc.reply('지금 모하는 거니?', 'Sadness'))
  print(cc.reply('지금 모하는 거니?', 'Happiness'))
  print(cc.reply('지금 모하는 거니?', 'Fear'))
  print(cc.reply('지금 모하는 거니?', 'Disgusting'))
  print(cc.reply('지금 모하는 거니?', 'Anger'))
  print(cc.reply('지금 모하는 거니?', 'Neutral'))

  print(cc.reply('나는 홍길동 이야', 'Neutral'))
  print(cc.reply('나는 홍길동 이야', 'surprised'))
  print(cc.reply('나는 홍길동 이야', 'Sadness'))
  print(cc.reply('나는 홍길동 이야', 'Happiness'))
  print(cc.reply('나는 홍길동 이야', 'Fear'))
  print(cc.reply('나는 홍길동 이야', 'Disgusting'))
  print(cc.reply('나는 홍길동 이야', 'Anger'))
  print(cc.reply('나는 홍길동 이야', 'Neutral'))
