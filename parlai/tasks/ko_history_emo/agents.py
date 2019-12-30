# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from parlai.core.agents import create_task_agent_from_taskname
from collections import deque
from .build import build

import copy
import random
import os


def _path(opt, filtered):
  # Build the data if it doesn't exist.
  build(opt)
  dt = opt['datatype'].split(':')[0]
  return os.path.join(opt['datapath'], 'KoMultiEmo20190226', dt + filtered + '.txt')


def flatten(teacher, context_length=-1, include_labels=True):
  """Return a flattened version of a teacher's data where all episodes only
  have length one but contain the desired amount of context.

  If context_length is not -1, will use only that many past utterances.
  Default is -1. Setting it to one only uses the input text.

  If include_labels is True, will include a random label in past utterances.
  Default is True.
  """
  data = []
  current = []
  episode_done = False
  context_length = context_length if context_length >= 0 else None
  context = deque(maxlen=context_length)
  try:
    while not teacher.epoch_done():
      # collect examples in episode
      while not episode_done:
        action = teacher.act()
        current.append(action)
        episode_done = action['episode_done']

      # build separate episodes from each example
      for ex in current:
        context.append(ex.get('text', ''))
        if len(context) > 1:
          ex.force_set('text', '\n'.join(context))
        ex.force_set('episode_done', True)
        if include_labels:
          # add labels to context
          labels = ex.get('labels', ex.get('eval_labels'))
          if labels is not None:
            context.append(random.choice(labels))
        data.append(ex)
      # reset flags and content
      episode_done = False
      current.clear()
      context.clear()
    return data
  except MemoryError as ex:
    raise MemoryError('Ran out of memory building flattened data batches. '
                      'Try using --context-length set to a small value to '
                      'limit the length of each flattened example, '
                      'disabling batch sorting / flattening by setting '
                      '--batch-sort false, or switching to data streaming '
                      'using --datatype {type}:stream to read from disk '
                      'if it is supported for your dataset.')


def make_batches(data, bsz):
  """Return a list of lists of size bsz given a list of examples."""
  return [data[i:i + bsz] for i in range(0, len(data), bsz)]


class DefaultTeacher(FbDialogTeacher):
  def __init__(self, opt, shared=None):
    opt = copy.deepcopy(opt)
    opt['datafile'] = _path(opt, '')
    if not opt['datatype'].startswith('train'):
      opt['cands_datafile'] = opt['datafile']
      if opt['batchsize'] > 1:
        if shared:
          if 'flatdata' in shared:
            self.flatdata = shared['flatdata']
        else:
          dt = opt.get('datatype', '').split(':')
          ordered_opt = opt.copy()
          ordered_opt['datatype'] = ':'.join((dt[0], 'ordered'))
          ordered_opt['batchsize'] = 1
          ordered_opt['numthreads'] = 1
          ordered_teacher = create_task_agent_from_taskname(ordered_opt)[0]
          clen = opt.get('context_length', 5)
          incl = opt.get('include_labels', True)
          self.flatdata = flatten(
              ordered_teacher, context_length=clen, include_labels=incl)
          self.batches = make_batches(self.flatdata, opt['batchsize'])
        self.flatdata_idx = 0
    opt['datatype'] = opt['datatype'].replace(':stream', '')
    super().__init__(opt, shared)
    if not opt['datatype'].startswith('train') and opt['batchsize'] > 1:
      self.use_batch_act = True
      self.lastY = None
      self.lastYs = [None] * opt['batchsize']

  def act(self):
    if self.opt['batchsize'] > 1:
      self.flatdata_idx += 1
      if self.flatdata_idx == len(self.flatdata):
        self.epochDone = True
      return self.flatdata[self.flatdata_idx - 1]
    else:
      return super().act()

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
