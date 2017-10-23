#!/bin/bash
exp_dir='exp-opensub'
#emb='data/glove.840B.300d.txt'
exp=
gpuid= 
model='seq2seq_v2'
emb=300
hs=1024
lr=0.0001
wd=0 #.00002
attn=false #true #False #True
attType=concat  #general concat dot

############### CUSTOM
gradClip=-1

tag='-bs128'
###############


train=1 # train=1, eval=0
OPTIND=1
while getopts "e:g:t:m:h:b:l:a:w:z:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
		m) model=$OPTARG ;;
		b) embsize=$OPTARG ;;
		h) hs=$OPTARG ;;
		l) lr=$OPTARG ;;
		a) attnType=$OPTARG ;;
		w) attn=$OPTARG ;;
		z) tag=$OPTARG;;
	esac
done
shift $((OPTIND -1))

exp=emb${emb}-hs${hs}-lr${lr}
if $attn ; then
	exp=$exp'-a_'${attType}
fi

if [ $(awk 'BEGIN{ print ('$wd' > '0') }') -eq 1 ]; then
	exp=$exp'-wd_'${wd}
fi


exp=${exp}${tag}

### '-' options are defined in parlai/core/params.py 
### -m --model : should match parlai/agents/<model> (model:model_class)
### -mf --model-file : model file name for loading and saving models
### 


if [ $train -eq 1 ]; then # train
	script='examples/train_model_seq2seq_ldecay.py'
	script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'.log'
	script=${script}' -bs 32' # training option
	script=${script}' -vparl 34436 -vp 5' #validation option
	#script=${script}' -vparl 100 -vp 10' #validation option
	script=${script}' -vmt nll -vme -1' #validation measure
	script=${script}' --optimizer adam -lr '${lr}
	
	#Dictionary arguments
	script=${script}' -dbf True --dict-file exp-opensub/dict_file_th5.dict' # built dict (word)
	script=${script}' --dict-minfreq 5'

	#seq2seq archituecture
#	script=${script}' -hs 2048 -emb 300 -dr 0.5 -att 0'

fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model.py'
	script=${script}' --datatype valid'
fi

#script=${script}' --embedding_file '$emb #validation option

if [ ! -d ${exp_dir}/exp-${exp} ]; then
	mkdir ${exp_dir}/exp-${exp}
fi

script=${script}' -m '${model}' -t opensubtitles -mf '${exp_dir}/exp-${exp}/exp-${exp}

if [ -n "$gpuid" ]; then
	script=${script}' --gpu '${gpuid}
fi

python ${script} -hs ${hs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd}


case "$exp" in
	e300-h2048) python ${script} -hs 1024 -emb 300 -att 0
		;;
esac

:<< comment
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

comment

