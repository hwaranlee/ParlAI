# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.
#import pandas as pd
import parlai.core.build_data as build_data
import xml.etree.ElementTree as etree

import html2text
import gzip
import os
import re
import codecs
import csv
import time

from konlpy.tag import Komoran
from konlpy.tag import Kkma
from examples.bot import Bot
from openpyxl import load_workbook

kkma = Kkma()

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

  emotion_map = {
      '중립': 'Neutral',
      '행복': 'Happiness',
      '슬픔': 'Sadness',
      '공포': 'Fear',
      '혐오': 'Disgust',
      '분노': 'Anger',
      '놀람': 'surprised'
  }

  templates = {}
  allowed_tags = []
  allowed_self_closing_tags = []
  allowed_attributes = []
  interwiki = {}
  namespaces = {}

  from mediawiki_parser.preprocessor import make_parser
  preprocessor = make_parser(templates)

  from mediawiki_parser.html import make_parser
  parser = make_parser(allowed_tags, allowed_self_closing_tags,
                       allowed_attributes, interwiki, namespaces)

  h = html2text.HTML2Text()
  h.ignore_links = True

  def parse(text):
    try:
      return h.handle(parser.parse(preprocessor.parse(text).leaves()).leaves())
    except:
      print('{} is not parsed well.'.format(text))
      return ''

  PATH_WIKI_XML = inpath
  FILENAME_WIKI = 'kowiki-latest-pages-articles.xml'
  ENCODING = "utf-8"

  def strip_tag_name(t):
    t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
      t = t[idx + 1:]
    return t

  pathWikiXML = os.path.join(PATH_WIKI_XML, FILENAME_WIKI)

  totalCount = 0
  articleCount = 0
  redirectCount = 0
  templateCount = 0
  title = None

  for event, elem in etree.iterparse(pathWikiXML, events=('start', 'end')):
    tname = strip_tag_name(elem.tag)

    if event == 'end':
      if tname == 'text':
        text = parse(elem.text)
        text = re.sub(r'\*\*(.*?)\*\*', '\\1', text)
        text = re.sub(r'{\|.*?\|}', '', text, flags=re.S)
        text = re.sub(r'<ref>.*?</ref>', '', text, flags=re.S)
        text = re.sub(r'<div.*?>|</div>', '\n\n', text, flags=re.S)
        text = re.sub(r'<br.*?>', '\n\n', text, flags=re.S)
        text = '\n'.join([t.replace('\n', ' ')
                          for t in text.split('\n\n')]).strip()
        for conv_id, paragraph in enumerate(text.split('\n')):
          line_id = 1
          dialog = ''
          paragraph = paragraph.strip()
          if paragraph:
            sentences = kkma.sentences(paragraph)
            if len(sentences) > 1:
              for turn_id, sentence in enumerate(sentences):
                value = preprocess(sentence) + ' Neutral'
                if turn_id % 2:
                  dialog += '\t{}\n'.format(value)
                  line_id += 1
                else:
                  dialog += '{} {}'.format(line_id, value)
              if dialog:
                handle = ftrain
                if conv_id % 10 == 0:
                  handle = ftest
                elif conv_id % 10 == 1:
                  handle = fvalid
                handle.write(dialog + '\n')

      elem.clear()

  ftrain.close()
  fvalid.close()
  ftest.close()


def build(opt):
  dpath = os.path.join(opt['datapath'], 'kowiki20190401')
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
