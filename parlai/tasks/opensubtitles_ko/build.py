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
    print(sent)
    return nlg.reply(sent)

def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    conv_id = 0
    # find all the files.
    for root, _subfolder, files in os.walk(inpath):
        for f in files:
            if root.endswith('ko/2016/5833686') and f == '6769905.xml.gz':
                continue

            if f.endswith('.gz'):
                dialog = ''
                conv_id = conv_id + 1
                with gzip.open(os.path.join(root, f), 'r') as f1:
                    # print(str(conv_id) + ': ' + f)
                    words = ''
                    line_id = 1
                    turn_id = 1
                    for line in f1:
                        line=line.decode('utf-8')
                        if re.search('<s .*id="', line):
                            # new sentence
                            if len(words) > 0:
                                if (turn_id % 2) == 0:
                                    dialog += str(line_id) + ' ' + preprocess(words)
                                else:
                                    dialog += '\t' + preprocess(words) + '\n'
                                    line_id += 1
                            turn_id = turn_id + 1
                            words = ''
                        else:
                            if re.search('<w .*id="', line):
                                word = line[line.find('>')+1:line.find('</w')]
                                words = words + ' ' + word.replace('\t', ' ')
                handle = ftrain
                if (conv_id % 10) == 0:
                    handle = ftest
                if (conv_id % 10) == 1:
                    handle = fvalid
                
                handle.write(dialog + '\n')
                

    ftrain.close()
    fvalid.close()
    ftest.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'OpenSubtitlesKo')
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

        create_fb_format(os.path.join(dpath, 'xml', 'ko'), dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
