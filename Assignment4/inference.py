import os
import sys
import json
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenize(sentence, word2token):
    tokenized = []
    for w in sentence.lower().split():
        token_id = word2token.get(w)
        if token_id is None:
            tokenized.append(word2token['<unk>'])
        else:
            tokenized.append(token_id)
    return tokenized

def main(text_path, model_code):
    maxlen = 30
    # load the two crucial dictionaries
    with open(os.path.join('data', 'word2token.json')) as f:
        word2token = json.load(f)
    with open(os.path.join('data', 'token2word.json')) as f:
        token2word = json.load(f)
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model(
        os.path.join('data/processed/{}.model'.format(model_code)))
    print(model.summary())

    with open(text_path) as f:
        sample_text = f.readlines()

    sample_text = [w.strip() for w in sample_text]
    sample_vec = []
    for sent in sample_text:
        sample_vec.append(tokenize(sent,word2token)) 

    print(sample_vec)
    sample_vec = pad_sequences(sample_vec, padding='post', maxlen=maxlen)
    print(sample_vec)

    print(model.predict(sample_vec))
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])