from parlai.core.dict import DictionaryAgent

class Dictionary(DictionaryAgent):
    def tokenize(self, text, building=False):
        return text.split()

