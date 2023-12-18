import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PreProcess:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.posTag = PosTag()
        self.lemmatizer = Lemmatizer()
        self.posMap = {
            'ADV': 1, 'CC': 2, 'DT': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'NEG': 7, 'NN': 8,
            'NNP': 9, 'NUM': 10, 'PR': 11, 'RP': 12, 'SC': 13, 'SYM': 14, 'UH': 15,
            'VB': 16, 'ADJP': 17, 'DP': 18, 'NP': 19, 'NUMP': 20, 'VP': 21
        }

        self.maxLen = 0
    
    def encode_pos_tag(self, pos):
        return self.posMap.get(pos, -1)

    def set_max_len(self, len):
        self.maxLen = len

    def text_to_sequence(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def pad_sequence_text(self, sequence):
        return pad_sequences(sequence, maxlen=self.maxLen)
    
    def extract_following_word(self, diWord):
        pattern = r'\bdi[-\s]*(.*?)(?=\s+di\b|$)'
        match = re.search(pattern, diWord, re.IGNORECASE)
        
        if match:
            following_phrase = match.group(1).strip()
            return following_phrase
        else:
            return None
    
    def extract_root_word(self, word):
        return self.lemmatizer.lemmatize(word)
    
    def get_pos_tag(self, word):
        if len(word.split()) > 1:
            pos_tags = self.posTag.get_phrase_tag(word)
        else:
            pos_tags = self.posTag.get_pos_tag(word)
        if pos_tags:
            return pos_tags[0][1]
        else:
            return None

    def start_pre_processing(self, data):
        self.tokenizer.fit_on_texts(pd.concat([data['word'], data['root_word'], data['following_word']]))
        word_sequences = self.text_to_sequence(data['word'])
        root_word_sequences = self.text_to_sequence(data['root_word'])
        following_word_sequences = self.text_to_sequence(data['following_word'])
        
        self.set_max_len(max(max(len(w) for w in word_sequences), max(len(r) for r in root_word_sequences), max(len(f) for f in following_word_sequences)))
