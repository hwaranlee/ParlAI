# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import gzip
import os
import re

from konlpy.tag import Komoran
from examples.bot import Bot

komoran = Komoran()
nlg = Bot('exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg'
        ,'exp-opensub_ko_nlg/dict_file_100000.dict', True)

def preprocess(sent):
    """ text preprocessing using a parser
    """
    return ' '.join(komoran.morphs(sent))

def postprocess(sent):
    sent = sent.replace(' __END__', '')
    sent = re.sub(' (.)$', '\\1', sent)
    return nlg.reply(sent)

def create_fb_format(inpaths, outpath):
    print('[building fbformat]')
    filenames = ['train.txt', 'valid.txt', 'test.txt']
    
    for fname in filenames:
        with open(os.path.join(outpath, fname), 'w') as outfile:
            for inpath in inpaths:
                with open(os.path.join(inpath, fname)) as infile:
                    for line in infile:
                        outfile.write(line)

def build(opt):
    inpaths = [os.path.join(opt['datapath'], 'AcrylKorean')
            , os.path.join(opt['datapath'], 'OpenSubtitlesKo')]
    outpath = os.path.join(opt['datapath'], 'KoMulti')
    version = None

    if not build_data.built(outpath, version_string=version):
        print('[building data: ' + outpath + ']')
        if build_data.built(outpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(outpath)
        build_data.make_dir(outpath)

        # Download the data.
        create_fb_format(inpaths, outpath)

        # Mark the data as built.
        build_data.mark_done(outpath, version_string=version)
