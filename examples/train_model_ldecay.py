# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Train a model.

After training, computes validation and test error.

Run with, e.g.:

python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf '/tmp/model'

..or..

python examples/train_model.py -m rnn_baselines/seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128

..or..

python examples/train_model.py -m drqa -t babi:Task10k:1 -mf '/tmp/model' -bs 10

TODO List:
- More logging (e.g. to files), make things prettier.
'''

### This script is for trianing SQuAD with PQMN
### by KAIST, CNSL
### with Learning rate decay

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
import build_dict
import copy
import importlib
import math
import os
import spacy
import logging, sys

import pdb

def run_eval(agent, opt, datatype, still_training=False, max_exs=-1):
    ''' Eval on validation/test data. '''
    print('[ running eval: ' + datatype + ' ]')
    opt['datatype'] = datatype
    if opt.get('evaltask'):
        opt['task'] = opt['evaltask']

    valid_world = create_task(opt, agent)
    cnt = 0
    for _ in valid_world:
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            first_run = False
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if valid_world.epoch_done() or (max_exs > 0 and cnt > max_exs):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_world.shutdown()
    valid_report = valid_world.report()

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if still_training:
        return valid_report
    elif opt['model_file']:
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

def main():
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
                        type=float, default=100)
    train.add_argument('-vtim', '--validation-every-n-secs',
                        type=float, default=-1)
    train.add_argument('-vparl', '--validation-every-n-parleys',
                        type=float, default=-1)
        
    train.add_argument('-vme', '--validation-max-exs',
                        type=int, default=-1,
                        help='max examples to use during validation (default ' +
                             '-1 uses all)')
    train.add_argument('-vp', '--validation-patience',
                        type=int, default=8,
                        help=('number of iterations of validation where result '
                              + 'does not improve before we stop training'))
    train.add_argument('-dbf', '--dict-build-first',
                        type='bool', default=True,
                        help='build dictionary first before training agent')
    opt = parser.parse_args()
    
    # Set logging
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if 'log_file' in opt:
        logfile = logging.FileHandler(opt['log_file'], 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))
    
    # Possibly build a dictionary (not all models do this).
    if opt['dict_build_first'] and 'dict_file' in opt:
        if opt['dict_file'] is None and opt.get('model_file'):
            opt['dict_file'] = opt['model_file'] + '.dict'
        logger.info("[ building dictionary first... ]")
        #pdb.set_trace()
        build_dict.build_dict(opt)

    # Build character dictionary
    if opt['add_char2word'] and opt['dict_build_first'] and 'dict_char_file' in opt:
        if opt['dict_char_file'] is None and opt.get('model_file'):
            opt['dict_char_file'] = opt['model_file'] + '.dict.char'
        logger.info("[ building character dictionary first... ]")
        # len(batch[0]_pdb.set_trace()
        build_dict.build_dict_char(opt)

    # TDNN setting (if using char)
    if opt['add_char2word']:
        opt['kernels'] = ''.join(opt['kernels'])
        if isinstance(opt['kernels'], str):
               opt['kernels'] = eval(opt['kernels']) # convert string list of tuple --> list of tuple
        opt['embedding_dim_TDNN']=0
        for i, n in enumerate(opt['kernels']):
            opt['embedding_dim_TDNN'] += n[1]

        logger.info('TDNN embedding dim = %d' % (opt['embedding_dim_TDNN']))


    # Create model and assign it to the specified task
    #pdb.set_trace()

    agent = create_agent(opt)
    world = create_task(opt, agent)


    train_time = Timer()
    validate_time = Timer()
    log_time = Timer()
    logger.info('[ training... ]')
    parleys = 0
    total_exs = 0   
    max_exs = opt['num_epochs'] * len(world)
    max_parleys = math.ceil(max_exs / opt['batchsize'])
    best_accuracy = 0
    impatience = 0
    saved = False
    while True:
        world.parley()
        parleys += 1

        if opt['num_epochs'] > 0 and parleys >= max_parleys:
            logger.info('[ num_epochs completed: {} ]'.format(opt['num_epochs']))
            break
        if opt['max_train_time'] > 0 and train_time.time() > opt['max_train_time']:
            logger.info('[ max_train_time elapsed: {} ]'.format(train_time.time()))
            break
#        if opt['log_every_n_secs'] > 0 and log_time.time() > opt['log_every_n_secs']:
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

            if hasattr(train_report, 'get') and train_report.get('total'):
                total_exs += train_report['total']
                logs.append('total_exs:{}'.format(total_exs))

            # check if we should log amount of time remaining
            time_left = None
            if opt['num_epochs'] > 0:
                exs_per_sec =  train_time.time() / total_exs
                time_left = (max_exs - total_exs) * exs_per_sec
            if opt['max_train_time'] > 0:
                other_time_left = opt['max_train_time'] - train_time.time()
                if time_left is not None:
                    time_left = min(time_left, other_time_left)
                else:
                    time_left = other_time_left
            if time_left is not None:
                logs.append('time_left:{}s'.format(math.floor(time_left)))

            # join log string and add full metrics report to end of log
            log = '[ {} ] {}'.format(' '.join(logs), train_report)

            # print(log)
            logger.info(log)
            log_time.reset()
            
#            if (opt['validation_every_n_secs'] > 0 and
#                    validate_time.time() > opt['validation_every_n_secs']):
            if (opt['validation_every_n_parleys'] > 0 and parleys % opt['validation_every_n_parleys'] == 0):
                valid_report = run_eval(agent, opt, 'valid', True, opt['validation_max_exs'])
                if valid_report['accuracy'] > best_accuracy:
                    best_accuracy = valid_report['accuracy']
                    impatience = 0
                    logger.info('[ new best accuracy: ' + str(best_accuracy) +  ' ]')
                    world.save_agents()
                    saved = True
                    if best_accuracy == 1:
                        logger.info('[ task solved! stopping. ]')
                        break
                else:
                    # doc_reader.model.opt['learning_rate'] *= 0.5
                    opt['learning_rate'] *= 0.5
                    agent.model.set_lrate(opt['learning_rate'])
                    logger.info('[ Decrease learning_rate %.2e]' % opt['learning_rate'] )
                    impatience += 1
                    logger.info('[ did not beat best accuracy: {} impatience: {} ]'.format(
                            round(best_accuracy, 4), impatience))
            
                validate_time.reset()
                if opt['validation_patience'] > 0 and impatience >= opt['validation_patience']:
                    logger.info('[ ran out of patience! stopping training. ]')
                    break
                        
    world.shutdown()
    if not saved:
        world.save_agents()
    else:
        # reload best validation model
        agent = create_agent(opt)

    run_eval(agent, opt, 'valid')
    #run_eval(agent, opt, 'test')


if __name__ == '__main__':
    main()
