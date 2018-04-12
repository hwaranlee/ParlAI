from parlai.core.dict import DictionaryAgent

class Dictionary(DictionaryAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        print('Using ko_multi dictionary.')
        special_tokens = [self.null_token, self.end_token,
                self.unk_token, self.start_token]

        for index, token in enumerate(special_tokens):
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokenize(self, text, building=False):
        return text.split()

