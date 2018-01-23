#!/bin/bash
exp_dir='exp'
#emb='data/glove.840B.300d.txt'
exp=exp-parallel
gpuid=0
model='seq2seq_v2'
emb=800
hs=4096
lr=0.0001
dr=0.5
wd=0 #.00002
attn=false #true # true / fase
attType=concat  #general concat dot
enc=gru
dict_maxexs=0
dict_nwords=20000
no_cuda=False
split_gpus=True
lt=unique
bi=False

############### CUSTOM
gradClip=-1

tag='separate'  #'-gc0.5' #'-bs128' #'-bs128'
############### EVALUATION
beam_size=5 #set 0 for greedy search

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
		a) attn=$OPTARG ;;
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


exp=${exp}-${tag}

### '-' options are defined in parlai/core/params.py 
### -m --model : should match parlai/agents/<model> (model:model_class)
### -mf --model-file : model file name for loading and saving models

if [ $train -eq 1 ]; then # train
	script='examples/train_model_seq2seq_ldecay.py'
	script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'.log'
	script=${script}' -bs 100' # training option
	script=${script}' -vparl 34436 -vp 5' #validation option
	script=${script}' -vmt nll -vme -1' #validation measure
	script=${script}' --optimizer adam -lr '${lr}
        script=${script}' --dropout '${dr}
        script=${script}' -enc '${enc}
        script=${script}' --split-gpus '${split_gpus}
        script=${script}' -lt '${lt}
        script=${script}' -bi '${bi}
        if [ $no_cuda = 'True' ]; then
            script=${script}' --no-cuda '
        fi
	
	#Dictionary arguments
	#script=${script}' -dbf True --dict-minfreq 5'
        script=${script}' -dbf True --dict-maxexs '${dict_maxexs}
        script=${script}' --dict-nwords '${dict_nwords}
fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model_human.py'
	script=${script}' --datatype valid'
	script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'_eval.log'
	script=${script}' --beam_size '$beam_size
fi

script=${script}' --dict-file exp-opensub/dict_file_20000.dict' # built dict (word)

#script=${script}' --embedding_file '$emb #validation option

if [ ! -d ${exp_dir}/exp-${exp} ]; then
	mkdir -p ${exp_dir}/exp-${exp}
fi

script=${script}' -m '${model}' -t opensubtitles_trial -mf '${exp_dir}/exp-${exp}/exp-${exp}

if [ -n "$gpuid" ]; then
	script=${script}' --gpu '${gpuid}
fi

# python -u -m cProfile -s "tottime" ${script} -hs ${hs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd}
python -u ${script} -hs ${hs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd}


case "$exp" in
	e300-h2048) python ${script} -hs 1024 -emb 300 -att 0
		;;
esac


