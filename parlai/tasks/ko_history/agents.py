# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os


def _path(opt, filtered):
  # Build the data if it doesn't exist.
  build(opt)
  dt = opt['datatype'].split(':')[0]
  return os.path.join(opt['datapath'], 'KoMulti',
                      dt + filtered + '.txt')


class DefaultTeacher(FbDialogTeacher):
  def __init__(self, opt, shared=None):
    opt = copy.deepcopy(opt)
    opt['datafile'] = _path(opt, '')
    if not opt['datatype'].startswith('train'):
      opt['cands_datafile'] = opt['datafile']
    opt['datatype'] = opt['datatype'].replace(':stream', '')
    super().__init__(opt, shared)

  def setup_data(self, path):
    previous_entry = None
    for entry, new in super().setup_data(path):
      if not new:
        yield (previous_entry[1][0], [entry[0]]), False

      previous_entry = entry
      if entry[1] is not None:
        yield entry, new

  def label_candidates(self):
    return None

  @staticmethod
  def get_key(data):
    xlen = len(data[0][0].split())
    ylen = len(data[0][1][0].split())
    ylen = ylen if xlen % 2 == 0 else -ylen

    return (xlen, ylen)

  def sort_data(self):
    # Sort based on the number of words in sentences.
    self.data.data.sort(key=DefaultTeacher.get_key)
