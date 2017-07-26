#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0
debug=0
OPTIND=1
while getopts "e:g:t:d:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
		d) debug=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

### '-' options are defined in parlai/core/params.py 
### -m --model : should match parlai/agents/<model> (model:model_class)
### -mf --model-file : model file name for loading and saving models
### 

if [ $train -eq 1 ]; then # train
	script='examples/train_model_ldecay.py'
	script=${script}' --log_file '$exp_dir'/exp-'${exp}'.log'
	script=${script}' -bs 32' # training option
	script=${script}' -vtim 1200 -vp 5' #validation option
	script=${script}' -dbf True --dict-file exp-squad/dict_file.dict' # built dict (word)
    script=${script}' --dict-char-file exp-squad/dict_file.dict.char' # built dict (char)    
fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model.py'
	script=${script}' --datatype valid'
fi

if [ $debug -eq 0 ]; then # eval
    script=${script}' --embedding_file $emb' #validation option 
fi

if [ $debug -eq 1 ]; then # eval
    script=${script}' -vparl 1' #validation option 
fi


#script=${script}' -m drqa -t squad -mf '${exp_dir}/exp-${exp}
script=${script}' -m pqmn -t squad -mf '${exp_dir}/exp-${exp}

script=${script}' --gpu '${gpuid}

case "$exp" in
    debug-pqmn-char) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --add_char2word True
		;;
	debug-pqmn) python $script --dropout_rnn 0.3 --dropout_emb 0.3 
		;;
	debug) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000   ## For debug
		;;
	debug-pos-ner) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --use_pos true --use_ner true
esac

