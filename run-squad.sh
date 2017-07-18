#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0

OPTIND=1
while getopts "e:g:t:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

### '-' options are defined in parlai/core/params.py 
### -m --model : should match parlai/agents/<model>
### -mf --model-file : model file name for loading and saving models
### 

if [ $train -eq 1 ]; then # train
	script='examples/train_model.py'
	script=${script}' --log_file '$exp_dir'/exp'${exp}'.log'
	script=${script}' -bs 32' # training option
	script=${script}' --embedding_file '${emb}
	script=${script}' -vtim 1200 -vp 5' #validation option
	script=${script}' -dbf True --dict-file exp-squad/dict_file.dict' # built dict
fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model.py'
	script=${script}' --datatype valid'
fi

script=${script}' -m drqa -t squad -mf '${exp_dir}/exp-${exp}


case "$exp" in
	debug) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000   ## For debug
		;;
	debug-pos-ner) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --use_pos true --use_ner true
esac

