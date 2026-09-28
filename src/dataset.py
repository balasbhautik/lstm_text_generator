import os
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class Textpreprocessor:
    """
    Preprocessor raw text  for character or word-level LSTM text generation.
    """

    def __init__(self, sequence_length=50, level='character'):
        self.sequence_length = sequence_length
        self.level = level
        self.tokenizer =  Tokenizer(filters='', lower=True, char_level=(level == 'character'))
        self.vocab_size = 0

    def clean_text(self, text: str):
        """
        Converts text to lowercase and strips non-essential punctuation.
        """
        text = text.lower()
        # Keep spaces and standard alphanumeric characters
        text =  re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def prepare_sequences(self, text: str):
        """
        Tokenizen cleaned text and forms (x, y) sliding windows sequence pairs.
        """

        cleaned = self.clean_text(text)

        # Fit tokenizer on cleaned  corpus
        self.tokenizer.fit_on_texts([cleaned])
        self.vocab_size = len(self.tokenizer.word_index) + 1 # 0 is reverved for padding

        # Convert text to sequence of interger indices
        tokens = self.tokenizer.texts_to_sequences([cleaned]) [0]

        X, y= [], []
        for i in range(0, len(tokens) - self.sequence_length):
            X.append(tokens[i : i + self.sequence_length])
            y.append(tokens[i + self.sequence_length])

        return  np.array(X), np.array(y)
    
        

