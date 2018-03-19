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

komoran = Komoran()

def preprocess(sent):
    """ text preprocessing using a parser
    """
    print(sent)
    return ' '.join(komoran.morphs(sent))

def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    conv_id = 0
    # find all the files.

    filebody = pd.read_excel(inpath, header=1)
    nums = filebody.values[:, 0]
    texts = filebody.values[:, 1]
    emotion_set = filebody.values[:, 2]
    
    dialog = ''

    lineid = 1
    conv_id = 1

    for i in range(0, len(nums)-1):
        if nums[i] < nums[i+1]:
            dialog = str(lineid) + ' ' + preprocess(texts[i]) + ' ' + emotion_set[i] + '\t' + preprocess(texts[i+1]) + ' ' + emotion_set[i+1] 
            lineid += 1
        else :
            conv_id += 1
            lineid = 0

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

        create_fb_format(os.path.join(dpath, 'result_data_01.xls'), dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
