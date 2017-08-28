#!/bin/bash
exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0
OPTIND=1
while getopts "e:g:t:d:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
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
	script=${script}' -vparl 3368 -vp 10' #validation option
	script=${script}' -dbf True --dict-file exp-squad/dict_file.dict' # built dict (word)
    script=${script}' --dict-char-file exp-squad/dict_file.dict.char' # built dict (char)    
fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model.py'
	script=${script}' --datatype valid'
fi

script=${script}' --embedding_file '$emb #validation option

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
		;;
	debug-pos-ner-char) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --use_pos true --use_ner true --add_char2word true
		;;
	h15) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --use_pos true --use_ner true --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False
		;;
	h16) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --use_pos true --use_ner true --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --add_char2word true --kernels '[(1, 15), (2, 20), (3, 35), (4, 40), (5, 75), (6, 90)]' --nLayer_Highway 1
		;;
	h16-1) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --use_pos true --use_ner true --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --add_char2word true --kernels '[(1, 5), (2, 10), (3, 15), (4, 20), (5, 25), (6, 30)]' --nLayer_Highway 1
		;;
	h16-2) python $script --dropout_rnn 0.3 --dropout_emb 0.3 --use_pos true --use_ner true --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --add_char2word true --kernels '[(5, 200)]' --nLayer_Highway 1

esac

