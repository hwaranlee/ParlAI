from flask import Flask, request
from examples.bot import Bot
import jpype
import json

bot = Bot('exp/exp-emb200-hs4096-lr0.0001-emotional_hred/' +
          'exp-emb200-hs4096-lr0.0001-emotional_hred',
          'exp-opensub_kemo_20190226/dict_file_100000.dict', True)

emotion_map = {10001: 'Happiness', 10002: 'Anger', 10003: 'Disgust',
               10004: 'Fear', 10005: 'Neutral', 10006: 'Sadness',
               10007: 'Surprise'}


class Flow:
  def reply(self, uid, utext, situation, emotion):
    if utext == 'Fallback':
      return None
    else:
      return utext, emotion


flow = Flow()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def server():
  utext = request.args.get('utext')
  emotion = request.args.get('emotion')
  print(emotion)
  uid = request.args.get('uid')
  situation = request.args.get('situation')
  ret = flow.reply(uid, utext, situation, emotion)
  if not ret:
    jpype.attachThreadToJVM()
    ret_text, ret_emotion = bot.reply(utext, emotion_map[int(emotion)], uid)
    for key, emotion in emotion_map.items():
      if emotion == ret_emotion:
        ret_emotion = key
        break
    ret = (ret_text, ret_emotion)

  return json.dumps({'text': ret[0], 'emotion': ret[1]},
                    ensure_ascii=False).encode('utf8')
