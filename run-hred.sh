#!/bin/bash
exp_dir='exp'
gpuid=0,1,2,3
model='hred'
emb=200
hs=4096
chs=2048
psize=1024
lr=0.0001
dr=0.5
wd=0 #.00002
attn=false #true # true / fase
attType=concat  #general concat dot
enc=gru
dict_maxexs=0
dict_maxtokens=100000
no_cuda=False
lt=unique
bi=False
embed='data/word2vec_ko/ko.bin'
dict_dir='exp-ko_multi_20180330'
dict_class='parlai.tasks.ko_multi.dict:Dictionary'
context_length=2
include_labels=False
pretrained_exp=exp-emb200-hs4096-lr0.0001-psize2_1024
pretrained_model_file='' #${exp_dir}/${pretrained_exp}/${pretrained_exp}

############### CUSTOM
gradClip=-1

tag='new_zeros1'  #'-gc0.5' #'-bs128' #'-bs128'
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
    z) tag=$OPTARG ;;
  esac
done
shift $((OPTIND -1))

exp=emb${emb}-hs${hs}-lr${lr}-chs${chs}
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
  script='examples/train_model_hred.py'
  script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'.log'
  script=${script}' -bs 100' # training option
  script=${script}' -vparl 30000 -vp 5' #validation option
  script=${script}' -vmt nll -vme -1' #validation measure
  script=${script}' --optimizer adam -lr '${lr}
  script=${script}' --dropout '${dr}
  script=${script}' -enc '${enc}
  script=${script}' -lt '${lt}
  script=${script}' -bi '${bi}
  script=${script}' --dict-class '${dict_class}
  script=${script}' --context-length '${context_length}
  script=${script}' --include-labels '${include_labels}
  script=${script}' --psize '${psize}
  if [ $no_cuda = 'True' ]; then
    script=${script}' --no-cuda'
  fi
  if [ $embed ]; then
    script=${script}' --embed '${embed}
  fi
  if [ $pretrained_model_file ]; then
    script=${script}' --pretrained_model_file '$pretrained_model_file
  fi

  #Dictionary arguments
  script=${script}' -dbf True --dict-maxexs '${dict_maxexs}
  script=${script}' --dict-maxtokens '${dict_maxtokens}
fi

if [ $train -eq 0 ]; then # eval
  script='examples/eval_model_human.py'
  script=${script}' --datatype valid'
  script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'_eval.log'
  script=${script}' --beam_size '$beam_size
  script=${script}' -bi '${bi}
  script=${script}' --optimizer adam -lr '${lr}
  script=${script}' -lt '${lt}
  script=${script}' --dict-class '${dict_class}
fi

mkdir -p $dict_dir
script=${script}' --dict-file '$dict_dir'/dict_file_'${dict_maxtokens}'.dict' # built dict (word)

if [ ! -d ${exp_dir}/exp-${exp} ]; then
  mkdir -p ${exp_dir}/exp-${exp}
fi

script=${script}' -m '${model}' -t ko_history -mf '${exp_dir}/exp-${exp}/exp-${exp}

if [ -n "$gpuid" ]; then
  script=${script}' --gpu '${gpuid}
fi

# python -u -m cProfile -s "tottime" ${script} -hs ${hs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd}
python -u ${script} -hs ${hs} -chs ${chs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd}


case "$exp" in
  e300-h2048) python ${script} -hs 1024 -emb 300 -att 0
    ;;
esac
