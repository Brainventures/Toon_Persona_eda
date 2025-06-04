import re
from konlpy.tag import Okt

def normalize_caption(text):
    # 특수 문자 및 불필요한 문자 제거
    text = re.sub(r"[^가-힣0-9.,!? ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
# oct = Okt()

# def clean_korean_text(text):
#     text = re.sub(r"[^가-힣0-9.,!? ]+", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def normalize_caption(text):
#     text = clean_korean_text(text)
#     tokens = oct.morphs(text)
#     return " ".join(tokens)

# def normalize_caption(text):
#     text = re.sub(r"\s+", " ", text.strip())          # 공백 정리
#     text = re.sub(r"([.!?])", r" \1", text)           # 문장 부호 분리
#     return text

class Vocab:
    def __init__(self):
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.word2count = {}
        self.nwords = 2

    def build_vocab(self, sentence):
        for word in sentence.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.nwords
                self.index2word[self.nwords] = word
                self.word2count[word] = 1
                self.nwords += 1
            else:
                self.word2count[word] += 1
