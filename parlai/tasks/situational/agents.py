# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os
import random

def _path(opt, is_domain):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train' or dt == 'test' or dt == 'valid':
        if is_domain:
            path = os.path.join(opt['datapath'], 'Situational', 'train_domain.txt')
        else:
            path = os.path.join(opt['datapath'], 'Situational', dt + '.txt')
    else:
        raise RuntimeError('Not valid datatype.')

    return path

class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if 'datafile' not in opt:
            opt['datafile'] = _path(opt, False)
            opt['datafile_domain'] = _path(opt, True)
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

        self.accuracy = 0
        self.total_error = 0
        self.ki = 0.01
        self.reduce_total_error = 0.9

        if 'ordered' not in opt['datatype'] and \
                opt['datafile'] != opt['datafile_domain'] and \
                shared == None and \
                opt['datatype'].startswith('train'):
            domain_opt = copy.deepcopy(opt)
            domain_opt['datafile'] = domain_opt['datafile_domain']
            self.domain_teacher = DefaultTeacher(domain_opt)

    def report_accuracy(self, accuracy):
        self.accuracy = accuracy
        self.total_error += 0.33 - accuracy
        if accuracy > 0.33:
            self.total_error *= self.reduce_total_error

    def setup_data(self, path):
        def rebuild(entries):
            return [(entries[i][1][0],
                [entries[i+1][0]]) for i in range(len(entries) - 1)]

        # this shows conversations in both directions
        alternate = []
        for entry, new in super().setup_data(path):
            if new:
                for i, e in enumerate(rebuild(alternate)):
                    yield e, i == 0
                alternate.clear()
            else:
                alternate.append(entry)
            yield entry, new
        if alternate:
            for i, e in enumerate(rebuild(alternate)):
                yield e, i == 0

    def label_candidates(self):
        return None

    def next_batch(self):
        if hasattr(self, 'domain_teacher'):
            t = len(self.domain_teacher.sorted_data)
            b = len(self.sorted_data)
            bsz = max(int(
                float(b) / (t + b) * self.accuracy * self.bsz - \
                        self.total_error * self.ki), 0)
            domain_bsz = self.bsz - bsz
    
            return self.next_batch_with_size(bsz) + \
                    self.domain_teacher.next_batch_with_size(domain_bsz)
        else:
            return self.next_batch_with_size(self.bsz)
 
    def next_batch_with_size(self, bsz):
        if self.index.value == -1:
            self.index.value = 0
        with self._lock():
            if self.index.value >= len(self.sorted_data):
                return [{'episode_done': True, 'id': self.getID()}] * self.bsz
            elif self.index.value + bsz >= len(self.sorted_data):
                self.epochDone = True
                if self.training:
                    front = self.sorted_data[self.index.value:]
                    if self.random:
                        random.shuffle(self.sorted_data)
                    self.index.value = (
                            self.index.value + bsz) % len(self.sorted_data)
                    back = self.sorted_data[:self.index.value]
                    return front + back
                else:
                    ret = self.sorted_data[self.index.value:]
                    self.index.value += bsz
                    return ret 
            else:
                self.epochDone = False
                ret = self.sorted_data[self.index.value:self.index.value + bsz]
                self.index.value += bsz
                return ret

