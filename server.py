from flask import Flask, request
from examples.bot import Bot
import jpype

bot = Bot('exp/exp-emb200-hs4096-lr0.0001-emotional_hred/' +
          'exp-emb200-hs4096-lr0.0001-emotional_hred',
          'exp-opensub_kemo_20190226/dict_file_100000.dict', True)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def server():
  jpype.attachThreadToJVM()
  s = request.args.get('s')
  print(s)
  e = request.args.get('e')
  i = request.args.get('i')

  return str(bot.reply(s, e, i))
