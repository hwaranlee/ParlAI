# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.
#import pandas as pd
import parlai.core.build_data as build_data
import gzip
import os
import re

from konlpy.tag import Komoran
from examples.bot import Bot
from openpyxl import load_workbook


komoran = Komoran()
nlg = Bot('exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg',
          'exp-opensub_ko_nlg/dict_file_100000.dict', True)


def preprocess(sent):
  """ text preprocessing using a parser
  """
  return ' '.join(komoran.morphs(sent))


def postprocess(sent):
  sent = re.sub(' __END__.*', '', sent)
  sent = re.sub('^- ', '', sent)
  sent = re.sub(' (.)$', '\\1', sent)
  wordlist = sent.split()
  if wordlist[-1] in ('Happiness', 'Neutral', 'Anger', 'Disgust', 'Sadness', 'surprised', 'Fear'):
    sent = ' '.join(wordlist[:-1])
  return nlg.reply(sent) + ' ' + wordlist[-1]
  # return wordlist[0] + ' ' + nlg.reply(sent)


def create_fb_format(inpath, outpath):
  print('[building fbformat]')
  ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
  fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
  ftest = open(os.path.join(outpath, 'test.txt'), 'w')

  conv_id = 0
  dialog = None
  dialogtemp = None
  # find all the files.

  emotion_map = {
      '중립': 'Neutral',
      '행복': 'Happiness',
      '슬픔': 'Sadness',
      '공포': 'Fear',
      '혐오': 'Disgust',
      '분노': 'Anger',
      '놀람': 'surprised'
  }

  for root, _subfolder, files in os.walk(inpath):
    for f in files:
      if f.endswith('.xlsx'):
        turn_id = None
        wb = load_workbook(os.path.join(root, f))
        ws = wb.active
        for row_idx, row in enumerate(ws.rows):
          preSentence = ''
          if row_idx == 0:
            continue

          if row[0].value == "S":
            if dialog:
              handle = ftrain
              if conv_id % 10 == 0:
                handle = ftest
              elif conv_id % 10 == 1:
                handle = fvalid
              handle.write(dialog + '\n')
            conv_id = conv_id + 1
            dialog = ''
            line_id = 1
            turn_id = 0

          if not turn_id is None:
            row2_value = row[2].value
            try:
              row2_value = emotion_map[row2_value]
            except:
              pass
            value = preprocess(row[1].value) + ' ' + row2_value
            #value = row[2].value + ' ' + preprocess(row[1].value)
            if turn_id % 2 == 0:
              preSentence = row[1].value
              dialogtemp = '{} {}'.format(line_id, value)
              turn_id += 1
            else:
              if(preSentence != row[1].value):
                dialogtemp += '\t{}\n'.format(value)
                line_id += 1
                turn_id += 1
                dialog += dialogtemp

          # if dialog != '':
          #    handle = ftrain
          #    if conv_id % 10 == 0:
          #        handle = ftest
          #    elif conv_id % 10 == 1:
          #        handle = fvalid
          #    handle.write(dialog + '\n')

  ftrain.close()
  fvalid.close()
  ftest.close()


def build(opt):
  dpath = os.path.join(opt['datapath'], 'KoMultiEmo20190226')
  version = None

  if not build_data.built(dpath, version_string=version):
    print('[building data: ' + dpath + ']')
    if build_data.built(dpath):
      # An older version exists, so remove these outdated files.
      build_data.remove_dir(dpath)
    build_data.make_dir(dpath)

    create_fb_format(dpath, dpath)

    # Mark the data as built.
    build_data.mark_done(dpath, version_string=version)
