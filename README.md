## Requirements
You may be able to use the library without this requirements, but the library is tested only from the below environment.
- One or more GeForce GTX 1080 Ti graphics cards (driver should be installed)
- CentOS 7

## Installation
1. Install bzip2  
```bash
sudo yum install bzip2
```
Follow instructions.


2. Install Anaconda with Python 3.6  
```bash
cd ~
mkdir Downloads
cd ~/Downloads
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
```
Follow instructions.  
Answer 'yes' to prepend the Anaconda3 install location to PATH in your .bashrc.  
Answer 'no' not to install Microsoft VSCode.


3. Install NVIDIA CUDA Toolkit 9.1
```bash
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/1/cuda_9.1.85.1_linux
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/2/cuda_9.1.85.2_linux
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda_9.1.85.3_linux
sudo sh cuda_9.1.85_387.26_linux
```
Follow instructions.  
Do not install NVIDIA Accelerated Graphics Driver, if you have already installed it.  
```bash
sudo sh cuda_9.1.85.1_linux
sudo sh cuda_9.1.85.2_linux
sudo sh cuda_9.1.85.3_linux
```
Follow instructions.


4. Install PyTorch
```bash
conda install pytorch torchvision cuda91 -c pytorch
```
Follow instructions.


5. Install ParlAI
```bash
python setup.py develop
```


6. Copy files  
  - Bot  
cc/exp-ko_multi_20180316/dict_file_100000.dict  
cc/exp-opensub_ko_nlg/dict_file_100000.dict  
cc/exp/exp-emb200-hs2048-lr0.0001-multi2018_30000/exp-emb200-hs2048-lr0.0001-multi2018_30000  
cc/exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg  
data/word2vec_ko/ko.bin  
  - Emotional Bot  
cc/exp-opensub_kemo_all/dict_file_100000.dict  
cc/exp-opensub_ko_nlg/dict_file_100000.dict  
cc/exp/exp-emb200-hs2048-lr0.0001-allK/exp-emb200-hs2048-lr0.0001-allK  
cc/exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg  
data/word2vec_ko/ko.bin
  - Context-Aware Emotional Bot  
cc/exp-opensub_kemo_20190226/dict_file_100000.dict  
cc/exp-opensub_ko_nlg/dict_file_100000.dict  
cc/exp/exp-emb200-hs2048-lr0.0001-mechanism/exp-emb200-hs2048-lr0.0001-mechanism  
cc/exp/exp-emb200-hs1024-lr0.0001-oknlg/exp-emb200-hs1024-lr0.0001-oknlg  
data/word2vec_ko/ko.bin


7. Install KoNLPy
```bash
sudo yum install gcc-c++ java-1.7.0-openjdk-devel python-devel
pip install JPype1-py3
pip install konlpy
```


8. Install Gensim
```bash
pip install gensim
```

## Usage
* Bot  

```python
from examples.bot import Bot

bot = Bot('exp/exp-emb200-hs2048-lr0.0001-multi2018_30000/exp-emb200-hs2048-lr0.0001-multi2018_30000', 'exp-ko_multi_20180316/dict_file_100000.dict', True)
answer = bot.reply('안녕')
```
* Emotional Bot  

```python
from examples.bot import Bot

bot = Bot('exp/exp-emb200-hs2048-lr0.0001-allK/exp-emb200-hs2048-lr0.0001-allK', 'exp-opensub_kemo_all/dict_file_100000.dict', True)
answer, emotion = bot.reply('안녕', 'Neutral') # The second parameter can be one of these: Neutral, Happiness, Anger, Sadness, Surprise, Fear, Disgust.
```
* Context-Aware Emotional Bot  

```python
from examples.bot import Bot

bot = Bot('exp/exp-emb200-hs2048-lr0.0001-mechanism/exp-emb200-hs2048-lr0.0001-mechanism', 'exp-opensub_kemo_20190226/dict_file_100000.dict', True)
answer, emotion = bot.reply('안녕', 'Neutral', 'test0') # The second parameter can be one of these: Neutral, Happiness, Anger, Sadness, Surprise, Fear, Disgust.
```
The last parameter of the Bot constructor determines whether to use CUDA or not.  
False is not tested.
