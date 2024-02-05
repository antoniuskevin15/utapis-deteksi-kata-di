import pandas as pd
import numpy as np
import re
import time
import warnings
import nltk
from tensorflow.keras.models import load_model

from Models.PreProcess import PreProcess

class DetectorModel:
    def __init__(self):
        self.preProcessingText = PreProcess()
        self.kata_dasar_di = pd.read_csv('./Data/kata_dasar_berawal_di.csv')
        self.nama_tempat = pd.read_csv('./Data/nama_tempat.csv')
        self.df = pd.read_csv('./Data/dataset_pos_tagged.csv')
        self.model = load_model('./Bot/di_detector_model.h5')
        self.start_pre_processing()
    
    def start_pre_processing(self):
        self.preProcessingText.start_pre_processing(self.df)

    def predict_di_word(self, word):
        # GET Following Word, Root Word, POS Tag
        word_pos = self.preProcessingText.get_pos_tag(word)

        following_word = self.preProcessingText.extract_following_word(word)
        following_word_pos = self.preProcessingText.get_pos_tag(following_word)
        
        root_word = self.preProcessingText.extract_root_word(word)
        root_word_pos = self.preProcessingText.get_pos_tag(root_word)

        # GET Words Sequence
        word_seq = self.preProcessingText.text_to_sequence([word])
        root_word_seq = self.preProcessingText.text_to_sequence([root_word])
        following_word_seq = self.preProcessingText.text_to_sequence([following_word])

        # GET Words Padded
        word_padded = self.preProcessingText.pad_sequence_text(word_seq)
        root_word_padded = self.preProcessingText.pad_sequence_text(root_word_seq)
        following_word_padded = self.preProcessingText.pad_sequence_text(following_word_seq)

        # GET POS Tag Encoded
        word_pos_encoded = self.preProcessingText.encode_pos_tag(word_pos)
        root_word_pos_encoded = self.preProcessingText.encode_pos_tag(root_word_pos)
        following_word_pos_encoded = self.preProcessingText.encode_pos_tag(following_word_pos)

        input_pos = np.array([[root_word_pos_encoded, following_word_pos_encoded, word_pos_encoded]])

        prediction = self.model.predict([word_padded, root_word_padded, following_word_padded, input_pos])

        return (prediction > 0.5).astype(int)[0][0]

    def detect_di_word(self, paragraph):
        di_pattern = r'\bdi[-\s]*\w+\b'
        prefix_pattern = r'\bdi[^\s]+\b'
        prefix__foreign_pattern = r'\bdi-[^\s]+\b'
        preposition_pattern = r'di\s+[^\s]+\b'

        result = {}

        sentences = nltk.sent_tokenize(paragraph)
        start_time = time.time()

        for sentence in sentences:
            di_words = re.finditer(di_pattern, sentence, re.IGNORECASE)

            for di_word in di_words:
                word = di_word.group()
                following_word = self.preProcessingText.extract_following_word(word)
                following_word_pos = self.preProcessingText.get_pos_tag(following_word)
                suggestion = None

                if word.lower() in self.kata_dasar_di['word'].values or following_word.lower() in self.nama_tempat['word'].values:
                    label = 1
                else:
                    label = self.predict_di_word(word)

                if label == 0:    
                    if following_word_pos.startswith('F'):
                        suggestion = f"di-{following_word}"
                    else:
                        root_word = self.preProcessingText.extract_root_word(following_word)

                        if re.search(prefix_pattern, word, re.IGNORECASE):
                            label = self.predict_di_word(f"di{root_word}")

                            if label == 0:
                                suggestion = f"di {following_word}"
                        elif re.search(preposition_pattern, word, re.IGNORECASE):
                            label = self.predict_di_word(f"di {root_word}")

                            if label == 0:
                                suggestion = f"di{following_word}"

                result[word] = {
                    'is_correct': True if label == 1 else False,
                    'suggestion': suggestion
                }

        end_time = time.time()
        print(f"Time Elapsed: {end_time - start_time}")

        return result
                