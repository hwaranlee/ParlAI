from examples.bot import Bot

bot = Bot('exp/exp-emb200-hs4096-lr0.0001-emotional_hred/' +
          'exp-emb200-hs4096-lr0.0001-emotional_hred',
          'exp-opensub_kemo_20190226/dict_file_100000.dict', True)

inpaths = ['data/KoMultiEmo20190226/train_multi_with_answer.txt',
           'data/KoMultiEmo20190226/test_multi_with_answer.txt',
           'data/KoMultiEmo20190226/valid_multi_with_answer.txt',
           'data/Script3/train_script_with_answer.txt',
           'data/Script3/test_script_with_answer.txt',
           'data/Script3/valid_script_with_answer.txt']

SEP = ' __SEP__ '


for inpath in inpaths:
  print(inpath)
  with open(inpath) as f:
    with open(inpath.replace('answer', 'prediction'), 'w') as out:
      if 'KoMultiEmo' in inpath:
        div = 10
      else:
        div = 1
      for idx, line in enumerate(f):
        if not idx % div:
          print(idx, end='\r')
          line = line.strip()
          line = line.split(SEP)[0]
          out.write(line + SEP +
                    bot.reply(line, 'Neutral', '{}{}'.format(inpath, idx))[0] +
                    '\n')
      print()
