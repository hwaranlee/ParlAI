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
from openpyxl import load_workbook

komoran = Komoran()

def preprocess(sent):
    """ text preprocessing using a parser
    """
    return ' '.join(komoran.morphs(sent))

def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    conv_id = 0
    dialog = None
    # find all the files.
    for root, _subfolder, files in os.walk(inpath):
        for f in files:
            if f.endswith('.xlsx'):
                wb = load_workbook(os.path.join(root, f))
                ws = wb.active
                for row_idx, row in enumerate(ws.rows):
                    if row_idx == 0 or row_idx == 1:
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

                    value = preprocess(row[1].value)
                    if turn_id % 2 == 0:
                        dialog += '{} {}'.format(line_id, value)
                    else:
                        dialog += '\t{}\n'.format(value)
                        line_id += 1

                    turn_id += 1

                if dialog != '':
                    handle = ftrain
                    if conv_id % 10 == 0:
                        handle = ftest
                    elif conv_id % 10 == 1:
                        handle = fvalid
                    handle.write(dialog + '\n')


    ftrain.close()
    fvalid.close()
    ftest.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'AcrylKorean')
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
