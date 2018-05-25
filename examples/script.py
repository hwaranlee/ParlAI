import os
import sys
import editdistance
import random

from openpyxl import load_workbook

class Bot:
    def __init__(self, inpath):
        self.data = {}

        for root, _, files in os.walk(inpath):
            for f in files:
                if f.endswith('.xlsx'):
                    wb = load_workbook(os.path.join(root, f))
                    for ws in wb:
                        for row_idx, row in enumerate(ws.rows):
                            if row_idx == 0:
                                continue

                            if row[1].value is None:
                                user_emotion = row[0].value
                            else:
                                try:
                                    pairs = self.data[user_emotion]
                                except KeyError:
                                    pairs = {}
                                    self.data[user_emotion] = pairs
                                
                                try:
                                    sentences = pairs[row[1].value]
                                except KeyError:
                                    sentences = []
                                    pairs[row[1].value] = sentences

                                sentences.append((row[3].value, row[2].value))
        
    def reply(self, message, *args):
        emotion = args[0]
        pairs = self.data[emotion]
        min_distance = sys.maxsize
        for sentence in pairs:
            distance = editdistance.eval(sentence, message)
            if distance < min_distance:
                min_distance = distance
                min_sentence = sentence

        return random.choice(pairs[min_sentence])

