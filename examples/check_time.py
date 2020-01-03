from examples.bot import Bot
from openpyxl import load_workbook
from datetime import datetime

import os
import time

print('{}: 학습 기반 대화 시스템 로딩 시작'.format(datetime.now()))
bot = Bot('exp/exp-emb400-hs2048-lr0.0001-transformer_40/exp-emb400-hs2048-lr0.0001-transformer_40',
          'exp-opensub_kemo_20190226/dict_file_10000.dict', True)
print('{}: 학습 기반 대화 시스템 로딩 완료'.format(datetime.now()))

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

batch_size = 4

for root, _subfolder, files in os.walk(inpath):
  for f in files:
    if f.endswith('.xlsx'):
      turn_id = None
      wb = load_workbook(os.path.join(root, f))
      ws = wb.active
      ds = []
      for row_idx, row in enumerate(ws.rows):
        if row_idx == 0:
          continue

        if row[0].value == "S":
          if dialog:
            if conv_id % 10 == 0:
              for idx, d in enumerate(dialog):
                start = time.time()
                if d[1] == 'surprised':
                  d = (d[0], 'Surprise')

                d = (d[0], d[1], str(conv_id) + '-' + str(idx % 2))

                ds.append(d)

                if len(ds) == batch_size:
                  for d in ds:
                    print('{}: {}, {} 입력'.format(datetime.now(), d[0], d[1]))
                  outputs = bot.batch_reply(ds)
                  for output in outputs:
                    print('{}: {}, {} 출력'.format(
                        datetime.now(), output[0], output[1]))

                  ds = []

                n += 1
                total_time += time.time() - start
                if n >= 1000:
                  print('{}: 총 걸린 시간: {}'.format(datetime.now(), total_time))
                  print('{}: 처리 문장 수: {}'.format(datetime.now(), n))
                  print('{}: 문장당 처리 시간: {}초'.format(
                      datetime.now(), total_time / n))
                  exit()
          conv_id += 1
          dialog = []
          turn_id = 0

        if not turn_id is None:
          row2_value = row[2].value
          try:
            row2_value = emotion_map[row2_value]
          except:
            pass
          value = (row[1].value, row2_value)
          turn_id += 1
          dialog.append(value)

print(total_time / n)
