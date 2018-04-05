import torch
import sys

from parlai.core.agents import create_agent
from parlai.core.worlds import validate
from parlai.core.params import ParlaiParser

import random
import logging, sys, os

class Bot:
    def __init__(self, model_path, dict_dir, cuda=False):
        opt = get_opt(model_path, cuda)
        opt['model_file'] = model_path        
        opt['datatype'] = 'valid'
        opt['dict_file'] = dict_dir
        opt['gpu'] = 0 if cuda else -1 # use cpu
        opt['cuda'] = cuda
        opt['no_cuda'] = (not cuda)
        opt['batchsize'] = 1

        self.opt = opt
        self.agent = create_agent(opt)        
        self.agent.training= False
        self.agent.generating = True
        
    def reply(self, message, *args):
        observation = {}
        message = self.agent.preprocess(message)
        observation['episode_done'] = True  ### TODO: for history
        if len(args) > 0:
            if args[0] == 'Surprise':
                emotion = 'surprised'
            else:
                emotion = args[0]
            observation['text'] = message + ' ' + emotion
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
            if emotion in ('Neutral', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Happiness', 'Disgust'):
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
    model_name = 'exp-emb300-hs1024-lr0.0001-bs128'
    pretrained_mdl_path = os.path.join(root_dir, 'exp-opensub/', model_name, model_name)  # release ver
    dict_dir = os.path.join(root_dir, 'exp-opensub/dict_file_th5.dict')

    cc = Bot(pretrained_mdl_path, dict_dir, cuda=False)

    # Example1 (in train)
    question_sample = "How many BS level degrees are offered in the College of Engineering at Notre Dame?"

    print(cc.reply(question_sample))
    print(cc.reply('How are you ?'))
    #print(cc.reply('nice to meet you'))
    #print(cc.reply(' don\'t be shy'))
    #print(cc.reply('welcome!!!!'))
 
