# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.
import pandas as pd
import parlai.core.build_data as build_data
import gzip
import os
import re

from konlpy.tag import Komoran
from examples.bot import Bot
from openpyxl import load_workbook


komoran = Komoran()
nlg = Bot('exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg'
        ,'exp-opensub_ko_nlg/dict_file_100000.dict', cuda=True)

def preprocess(sent):
    """ text preprocessing using a parser
    """
    return ' '.join(komoran.morphs(sent))

def postprocess(sent):
    sent = sent.replace(' __END__', '')
    sent = re.sub('^- ', '', sent)
    sent = re.sub(' (.)$', '\\1', sent)
    wordlist = sent.split()
    if wordlist[0] in ('Happiness', 'Neutral', 'Anger', 'Disgust', 'Sadness', 'surprised', 'Fear'):
        sent = ' '.join(wordlist[1:])
    return nlg.reply(sent) + ' ' + wordlist[0]

def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    conv_id = 0
    dialog = None
    dialogtemp = None
    # find all the files.

    for root, _subfolder, files in os.walk(inpath):
        for f in files:
            if f.endswith('.xlsx') :
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

                    if row[2].value == 'surprised':
                        row[2].value = 'Surprise'

                    value = row[2].value + ' ' + preprocess(row[1].value)
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

    ftrain.close()
    fvalid.close()
    ftest.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'KoreanWithEmotion')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        # url = ('http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz')
        # build_data.download(url, dpath, 'OpenSubtitles.tar.gz')
        # build_data.untar(dpath, 'OpenSubtitles.tar.gz', deleteTar=False)

        #create_fb_format(os.path.join(dpath, 'OpenSubwithemotion2018.csv'), dpath)
        create_fb_format(dpath, dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
