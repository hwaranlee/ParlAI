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
import random

def preprocess(sent):
    """ text preprocessing using a parser
    """
    return sent

def postprocess(sent):
    return sent

def write(source, target):
    if random.random() < write.train_ratio:
        out_file = write.ftrain
    else:
        out_file = write.fvalid
    out_file.write('1 ' + source + '\t' + target + '\n')

def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    filenames = ['train.txt', 'valid.txt', 'test.txt']

    write.ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    write.fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    write.train_ratio = 0.7

    out_index = 0
    for fname in filenames:
        with open(os.path.join(inpath, fname)) as infile:
            for line in infile:
                if line.strip() != '':
                    space_idx = line.find(' ')
                    script_idx = int(line[:space_idx])
                    if script_idx == 1:
                        sentences = [None, None]
                    line = line[line.find(' ') + 1 :].split('\t')
                    sentences[0] = line[0]

                    if sentences[1] is not None:
                        write(sentences[1], sentences[0])

                    if len(line) > 1:
                        sentences[1] = line[1].strip()
                        write(sentences[0], sentences[1])

    write.ftrain.close()
    write.fvalid.close()

def build(opt):
    inpath = os.path.join(opt['datapath'], 'OpenSubtitles')
    outpath = os.path.join(opt['datapath'], 'GoogleOpenSubtitles')
    version = None

    if not build_data.built(outpath, version_string=version):
        print('[building data: ' + outpath + ']')
        if build_data.built(outpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(outpath)
        build_data.make_dir(outpath)

        # Download the data.
        create_fb_format(inpath, outpath)

        # Mark the data as built.
        build_data.mark_done(outpath, version_string=version)
