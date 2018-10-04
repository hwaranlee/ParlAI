# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present
"""Train a model.
After training, computes validation and test error.
"""


import torch
import random
import os
import datetime

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.worlds import BatchWorld
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from examples.build_dict import setup_args as setup_dict_args
import build_dict
import math
import logging
import sys


def run_eval(agent, opt, datatype, max_exs=-1, write_log=False, valid_world=None, logger=None, generate=False, local_human=False):
  """Eval on validation/test data.
  - Agent is the agent to use for the evaluation.
  - opt is the options that specific the task, eval_task, etc
  - datatype is the datatype to use, such as "valid" or "test"
  - write_log specifies to write metrics to file if the model_file is set
  - max_exs limits the number of examples if max_exs > 0
  - valid_world can be an existing world which will be reset instead of reinitialized
  """

  if logger == None:
    print("please add logger")
    return

  logger.info('[ running eval: ' + datatype + ' ]')
  if 'stream' in opt['datatype']:
    datatype += ':stream'
  opt['datatype'] = datatype
  if opt.get('evaltask'):
    opt['task'] = opt['evaltask']

  if valid_world is None:
    valid_world = create_task(opt, agent)
  else:
    valid_world.reset()
  cnt = 0

  agent.training = False
  agent.local_human = local_human

  if generate:
    agent.generating = True
    print("Generating:")

  while not valid_world.epoch_done():
    if isinstance(valid_world, BatchWorld):
      valid_world.batch_observations[0] = None
    valid_world.parley()

    if cnt == 0 and opt['display_examples']:
      logger.info(valid_world.display() + '\n~~')
      logger.info(valid_world.report())
    cnt += opt['batchsize']
    if max_exs > 0 and cnt > max_exs + opt.get('numthreads', 1):
      # note this max_exs is approximate--some batches won't always be
      # full depending on the structure of the data
      break

  #valid_report = valid_world.report()
  if hasattr(agent, 'report'):
    valid_report = agent.report()
    agent.reset_metrics()
  else:
    valid_report = valid_world.report()
    valid_world.reset_metrics()

  metrics = datatype + ':' + str(valid_report)
  logger.info(metrics)
  if write_log and opt['model_file']:
    # Write out metrics
    f = open(opt['model_file'] + '.' + datatype, 'a+')
    f.write(metrics + '\n')
    f.close()

  agent.reset_valid_report()
  agent.training = True

  return valid_report, valid_world


def get_n_batches(world):
  return len(world.world.get_agents()[0].batches)


def main():
  torch.cuda.manual_seed_all(0)
  torch.manual_seed(0)
  random.seed(0)

  # Get command line arguments
  parser = ParlaiParser(True, True)
  train = parser.add_argument_group('Training Loop Arguments')
  train.add_argument('-et', '--evaltask',
                     help=('task to use for valid/test (defaults to the ' +
                            'one used for training if not set)'))
  train.add_argument('-d', '--display-examples',
                     type='bool', default=False)
  train.add_argument('-e', '--num-epochs', type=float, default=-1)
  train.add_argument('-ttim', '--max-train-time',
                     type=float, default=-1)
  train.add_argument('-ltim', '--log-every-n-secs',
                     type=float, default=2)
  train.add_argument('-lparl', '--log-every-n-parleys',
                     type=int, default=100)
  train.add_argument('-vtim', '--validation-every-n-secs',
                     type=float, default=-1)
  train.add_argument('-vparl', '--validation-every-n-parleys',
                     type=int, default=-1)
  train.add_argument('-vme', '--validation-max-exs',
                     type=int, default=-1,
                     help='max examples to use during validation (default ' +
                     '-1 uses all)')
  train.add_argument('-vp', '--validation-patience',
                     type=int, default=5,
                     help=('number of iterations of validation where result '
                            + 'does not improve before we stop training'))
  train.add_argument('-vmt', '--validation-metric', default='accuracy',
                     help='key into report table for selecting best validation')
  train.add_argument('-dbf', '--dict-build-first',
                     type='bool', default=True,
                     help='build dictionary first before training agent')
  train.add_argument('-logger', '--log-file', default='',
                     help='log file name')
  train.add_argument('-dn', '--dict-nwords', type=int,
                     default=1000000, help='The number of words for dictionary')
  train.add_argument('--split-gpus', type=bool, default=False,
                     help='Split gpus for a large model.')

  parser = setup_dict_args(parser)

  opt = parser.parse_args()
  model_save_path = opt.get('model_file', None)

  # Set logging
  logger = logging.getLogger('hred')
  logger.setLevel(logging.INFO)
  fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
  console = logging.StreamHandler()
  console.setFormatter(fmt)
  logger.addHandler(console)
  if 'log_file' in opt:
    if os.path.exists(os.path.dirname(opt['log_file'])):
      opt['log_file'] = opt['log_file'] + '_' + \
          datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
      os.makedirs(os.path.dirname(opt['log_file']), exist_ok=True)
    else:
      os.makedirs(os.path.dirname(opt['log_file']), exist_ok=True)
    logfile = logging.FileHandler(opt['log_file'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
  logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

  # Possibly build a dictionary (not all models do this).
  if opt['dict_build_first'] and 'dict_file' in opt:
    if opt['dict_file'] is None and opt.get('model_file'):
      opt['dict_file'] = opt['model_file'] + '.dict'
    logger.info("[ building dictionary first... ]")
    build_dict.build_dict(opt)

  # Create model and assign it to the specified task
  agent = create_agent(opt)
  world = create_task(opt, agent)

  train_time = Timer()
  validate_time = Timer()
  log_time = Timer()
  logger.info('[ training... ]')
  parleys = 0
  total_exs = 0
  max_parleys = opt['num_epochs'] * get_n_batches(world)
  impatience = 0
  saved = False
  valid_world = None
  best_loss = 1000000
  scheduler = ReduceLROnPlateau(
      agent.optimizer, factor=0.5, patience=1, verbose=True)

  def get_lr():
    return agent.optimizer.param_groups[0]['lr']

  while True:
    if not agent.training:
      agent.training = True

    world.parley()
    parleys += 1

    if opt['num_epochs'] > 0 and parleys >= max_parleys:
      logger.info('[ num_epochs completed: {} ]'.format(opt['num_epochs']))
      break
    if opt['max_train_time'] > 0 and train_time.time() > opt['max_train_time']:
      logger.info('[ max_train_time elapsed: {} ]'.format(train_time.time()))
      break

    # instead of every_n_secs, use n_parleys
    # if True:
    if opt['log_every_n_parleys'] > 0 and parleys % opt['log_every_n_parleys'] == 0:
      if opt['display_examples']:
        logger.info(world.display() + '\n~~')

      logs = []
      # time elapsed
      logs.append('time:{}s'.format(math.floor(train_time.time())))
      logs.append('parleys:{}'.format(parleys))

      # get report and update total examples seen so far
      if hasattr(agent, 'report'):
        train_report = agent.report()
        agent.reset_metrics()
      else:
        train_report = world.report()
        world.reset_metrics()

      # join log string and add full metrics report to end of log
      log = '[ {} ] {}'.format(' '.join(logs), train_report)

      logger.info(log)
      log_time.reset()

    if (opt['validation_every_n_parleys'] > 0 and parleys % opt['validation_every_n_parleys'] == 0):
      # if True :
      valid_report, valid_world = run_eval(
          agent, opt, 'valid', opt['validation_max_exs'], logger=logger)
      # if False :
      if valid_report[opt['validation_metric']] < best_loss:  # best_accuracy=nll
        best_loss = valid_report[opt['validation_metric']]
        impatience = 0
        logger.info('[ new best loss(nll): ' + str(best_loss) + ' ]')
        agent.save(model_save_path)
        saved = True
        if best_loss == 1:
          logger.info('[ task solved! stopping. ]')
          break
      # if True:
      else:
        impatience += 1
        logger.info('[ did not beat best accuracy: {} impatience: {} ]'.format(
            round(best_loss, 4), impatience))

      scheduler.step(valid_report[opt['validation_metric']])
      opt['learning_rate'] = get_lr()

      validate_time.reset()
      if opt['validation_patience'] > 0 and impatience >= opt['validation_patience']:
        logger.info('[ ran out of patience! stopping training. ]')
        break
      if get_lr() < pow(10, -6):
        logger.info('[ learning_rate < pow(10,-6) ! stopping training. ]')
        break

  world.shutdown()
  if not saved:
    world.save_agents()
  else:
    # reload best validation model
    agent = create_agent(opt)

  # run_eval(agent, opt, 'valid', write_log=True, logger=logger, generate=True)
  # run_eval(agent, opt, 'test', write_log=True, logger=logger, generate=True)


if __name__ == '__main__':
  main()
