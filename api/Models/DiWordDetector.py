import re
import nltk
import time
import warnings
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nlp_id.postag import PosTag
from nlp_id.lemmatizer import Lemmatizer 

warnings.filterwarnings("ignore")

class DiWordDetector:
    def __init__(self):
        self.postagger = PosTag()
        self.lemmatizer = Lemmatizer()

    def check_n_gram(self, sentence, di_word, n = 3):
        words = word_tokenize(sentence)
        prev_words = []
        next_words = []
        
        # Generate n-grams from the words
        ngrams_result = list(ngrams(words, n))

        # Filter n-grams that contain di_word
        matching_ngrams = [gram for gram in ngrams_result if di_word in gram]

        # Process the matching n-grams
        for gram in matching_ngrams:
            di_word_index = gram.index(di_word)
            
            # Process words before di_word
            for i in range(di_word_index):
                word = gram[i]
                if word not in prev_words:
                    prev_words.append(word)

            # Process words after di_word
            for i in range(di_word_index + 1, len(gram)):
                word = gram[i]
                if word not in next_words:
                    next_words.append(word)

        return list(prev_words), list(next_words)
    
    def check_word_usage_in_sentence(self, di_word, word, sentence_pos, sentence, is_prefix):
        if is_prefix:
            n = 3
            root_word = self.lemmatizer.lemmatize(word)
            root_word_pos = self.postagger.get_pos_tag(root_word)[0][1]

            check_word = di_word

            if root_word_pos.startswith('V'):
                return True
        else:
            n = 5
            check_word = word

        words_pos_in_sentence = {}
        for w, tag in sentence_pos:
            words_pos_in_sentence[w] = tag

        prev_words, next_words = self.check_n_gram(sentence, check_word, n)

        is_correct = not is_prefix

        if len(prev_words) == 0 and len(next_words) == 0:
            return True

        if len(next_words) == 1:
            if next_words[0] == '.' and not is_prefix:
                return not is_prefix

        for i in range(len(next_words)):
            if words_pos_in_sentence.get(next_words[i]) is None:
                if not self.postagger.get_pos_tag(next_words[i])[0][1].startswith('V'):
                    next_words[i] = self.lemmatizer.lemmatize(next_words[i])
            if next_words[i] == '':
                next_words[i] = '"'

        for i in range(len(prev_words)):
            if words_pos_in_sentence.get(prev_words[i]) is None:
                if not self.postagger.get_pos_tag(prev_words[i])[0][1].startswith('V'):
                    prev_words[i] = self.lemmatizer.lemmatize(prev_words[i])
            if prev_words[i] == '':
                prev_words[i] = '"'

        verb_exist_prev = False
        for i in range(len(prev_words)):
            pos_prev = words_pos_in_sentence[prev_words[i]] if words_pos_in_sentence.get(prev_words[i]) is not None else self.postagger.get_pos_tag(prev_words[i])[0][1]
            if not verb_exist_prev:
                if pos_prev.startswith('V'):
                    verb_exist_prev = True
                    is_correct = not is_prefix
                else:
                    is_correct = is_prefix
            elif len(prev_words) > 1:
                if pos_prev == 'CC':
                    is_correct = is_prefix
                elif pos_prev.startswith('V'):
                    is_correct = not is_prefix
            else:
                is_correct = not is_prefix
                
        verb_exist_next = False

        for i in range(len(next_words)):
            pos_next = words_pos_in_sentence[next_words[i]] if words_pos_in_sentence.get(next_words[i]) is not None else self.postagger.get_pos_tag(next_words[i])[0][1]
            if not verb_exist_next:
                if pos_next.startswith('V'):
                    verb_exist_next = True
                    is_correct = not is_prefix
                    if i == 1 and words_pos_in_sentence[next_words[i - 1]] == 'CC':
                        is_correct = is_prefix
                    else:
                        is_correct = not is_prefix
                else:
                    if not verb_exist_prev and i == len(next_words) - 1:
                        is_correct = is_prefix
                    else:
                        is_correct = not is_prefix
            else:
                if pos_next.startswith('V'):
                    is_correct = not is_prefix
                else:
                    is_correct = is_prefix

        if is_prefix:
            if not verb_exist_prev and not verb_exist_next:
                is_correct = is_prefix
        else:
            if verb_exist_prev or verb_exist_next:
                is_correct = not is_prefix

        return is_correct

    def extract_following_word(self, di_phrase):
        di_pattern = r'\bdi\s*(\w+)\b'
        match = re.search(di_pattern, di_phrase, re.IGNORECASE)
        if match:
            following_word = match.group(1)
            return following_word
        else:
            return None

    def detect_di_usage(self, paragraph):
        di_pattern = r'\bdi\s*\w+\b'

        di_results = {}

        # Tokenize the paragraph into sentences
        sentences = nltk.sent_tokenize(paragraph)

        for sentence in sentences:
            sentence_pos = self.postagger.get_pos_tag(sentence)
            # Find all occurrences of 'di' followed by a single word in each sentence
            di_matches = re.finditer(di_pattern, sentence, re.IGNORECASE)  

            for match in di_matches:
                # Get the di word (preposition and prefix)
                di_word = match.group()

                following_word = self.extract_following_word(di_word)

                combined_word = 'di' + following_word
                combined_word_tags = self.postagger.get_pos_tag(combined_word)
                combined_word_pos = combined_word_tags[0][1]

                # Define regular expression patterns for prefix and preposition
                prefix_pattern = r'\bdi[^\s]+\b'
                preposition_pattern = r'di\s+[^\s]+\b'

                is_correct = None
                suggestion = None

                # Check if 'di' matches the prefix pattern (imbuhan)
                if re.search(prefix_pattern, di_word, re.IGNORECASE):
                    # Perform part-of-speech tagging on the following word
                    pos_tags = self.postagger.get_pos_tag(following_word)

                    for word, pos in pos_tags:
                        if pos.startswith('V'):
                            # If the following word is a verb, consider it correct
                            is_correct = True
                            suggestion = None
                            break
                        elif pos.startswith('N'):
                            if combined_word_pos.startswith('N') or combined_word_pos.startswith('P'):
                                is_correct = True
                                suggestion = None
                            else:
                                is_correct = False
                                suggestion = f"di {word}"
                            if combined_word_pos.startswith('V'):
                                is_correct = self.check_word_usage_in_sentence(di_word, word, sentence_pos, sentence, True)
                                suggestion = None if is_correct else f"di {word}"
                                break
                        elif pos.startswith('D') or pos.startswith('I') or pos.startswith('A') or pos.startswith('P'):
                            is_correct = False
                            suggestion = f"di {word}"
                            break
                        elif pos.startswith('F') or combined_word_pos.startswith('F'):
                            is_correct = True
                            suggestion = None
                            break
                        elif not pos.startswith('N') and not pos.startswith('V'):
                            if combined_word_pos.startswith('V'):
                                is_correct = True
                                suggestion = None
                                break
                        else:
                            is_correct = False
                            suggestion = f"di {word}"
                            break

                # Check if 'di' matches the preposition pattern (kata depan)
                elif re.search(preposition_pattern, di_word, re.IGNORECASE):
                    # Perform part-of-speech tagging on the following word
                    pos_tags = self.postagger.get_pos_tag(following_word)

                    for word, pos in pos_tags:
                        if pos.startswith('N') or pos.startswith('A') or pos.startswith('F') or pos.startswith('D') or pos.startswith('I'):
                            is_correct = True
                            suggestion = None
                            
                            if pos.startswith('N') and combined_word_pos.startswith('V'):
                                if pos == 'NNP':
                                    is_correct = True
                                    break
                                is_correct = self.check_word_usage_in_sentence(di_word, word, sentence_pos, sentence, False)
                                suggestion = None if is_correct else f"di{word}"
                                break
                        elif pos.startswith('P'):
                            is_correct = True
                            break
                        else:
                            if combined_word_pos.startswith('V'):
                                # If the following word is a noun, consider it correct
                                is_correct = False
                                suggestion = f"{combined_word}"
                                break

                if is_correct is None:
                    # If neither prefix nor preposition pattern was matched, suggest a correction
                    is_correct = False
                    suggestion = f"Suggest your correction here for '{di_word}' (e.g., 'di {following_word}')."

                if di_word in di_results:
                    if di_results[di_word]['is_correct'] == False and is_correct == True:
                        di_results[di_word] = {
                            'is_correct': is_correct,
                            'suggestion': suggestion
                        }
                else:
                    di_results[di_word] = {
                        'is_correct': is_correct,
                        'suggestion': suggestion
                    }

        return di_results