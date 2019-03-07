from examples.bot import Bot
from openpyxl import load_workbook

import os
import time

bot = Bot('exp/exp-emb200-hs4096-lr0.0001-emotional_hred/exp-emb200-hs4096-lr0.0001-emotional_hred',
          'exp-opensub_kemo_20190226/dict_file_100000.dict', True)

inpath = 'data/KoMultiEmo20190226'

conv_id = 0
dialog = []
dialogtemp = None
# find all the files.

emotion_map = {
    '중립': 'Neutral',
    '행복': 'Happiness',
    '슬픔': 'Sadness',
    '공포': 'Fear',
    '혐오': 'Disgust',
    '분노': 'Anger',
    '놀람': 'surprised'
}

total_time = 0
n = 0

for root, _subfolder, files in os.walk(inpath):
  for f in files:
    if f.endswith('.xlsx'):
      turn_id = None
      wb = load_workbook(os.path.join(root, f))
      ws = wb.active
      for row_idx, row in enumerate(ws.rows):
        preSentence = ''
        if row_idx == 0:
          continue

        if row[0].value == "S":
          if dialog:
            if conv_id % 10 == 0:
              for idx, d in enumerate(dialog):
                start = time.time()
                bot.reply(d[0], d[1], str(conv_id) + '-' + str(idx % 2))
                n += 1
                total_time += time.time() - start
              if n >= 100:
                print(total_time / n)
                exit()
          conv_id += 1
          dialog = []
          line_id = 1
          turn_id = 0

        if not turn_id is None:
          row2_value = row[2].value
          try:
            row2_value = emotion_map[row2_value]
          except:
            pass
          value = (row[1].value, row2_value)
          # value = row[2].value + ' ' + preprocess(row[1].value)
          if turn_id % 2 == 0:
            preSentence = row[1].value
            dialogtemp = value
            turn_id += 1
          else:
            if(preSentence != row[1].value):
              line_id += 1
              turn_id += 1
              dialog.append(dialogtemp)

print(total_time / n)
