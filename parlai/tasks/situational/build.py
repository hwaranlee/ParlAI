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
nlg = None

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

def create_fb_format(domain_inpath, inpaths, outpath):
    print('[building fbformat]')
    filenames = ['train.txt', 'valid.txt', 'test.txt']

    for fname in filenames:
        with open(os.path.join(outpath, fname), 'w') as outfile:
            for inpath in inpaths:
                with open(os.path.join(inpath, fname)) as infile:
                    for line in infile:
                        outfile.write(line)

    ftrain = open(os.path.join(outpath, 'train_domain.txt'), 'w')

    conv_id = 0
    for root, _subfolder, files in os.walk(domain_inpath):
        for f in files:
            if f.endswith('.xlsx') :
                wb = load_workbook(os.path.join(root, f))
                for ws in wb:
                    for row_idx, row in enumerate(ws.rows):
                        if row_idx == 0 :
                            continue

                        if row[1].value is None:
                            user_emotion = row[0].value
                        else:
                            ftrain.write('1 {} {}\t{} {}\n'.format(
                                user_emotion, preprocess(row[1].value),
                                row[2].value, preprocess(row[3].value)))
                            conv_id = conv_id + 1
    
    ftrain.close()

def build(opt):
    nlg = Bot('exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg'
            ,'exp-opensub_ko_nlg/dict_file_100000.dict', True, opt['gpu'])

    inpaths = [os.path.join(opt['datapath'], 'KoreanWithEmotion')]
    dpath = os.path.join(opt['datapath'], 'Situational')
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
        create_fb_format(dpath, inpaths, dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
