import os
import sys
from pprint import pprint
import numpy as np
import json
import pickle
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import keras
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Flatten
from keras.regularizers import l2



def read_csv(data_dir):
    with open(data_dir) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]

def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test

def build_embedding_mat(data_dir, vocab, w2v):
    """
    Build the embedding matrix which will be used to initialize weights of
    the embedding layer in our seq2seq architecture
    """
    # we have 4 special tokens in our vocab
    token2word = {0: '<sos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
    word2token = {'<sos>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
    # +4 for the four vocab tokens
    embedding_dim=100
    vocab_size = len(vocab) + 4
    embedding_dim = embedding_dim
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    embedding_matrix[0] = np.random.random((1, embedding_dim))
    embedding_matrix[1] = np.random.random((1, embedding_dim))
    embedding_matrix[2] = np.random.random((1, embedding_dim))
    embedding_matrix[3] = np.random.random((1, embedding_dim))
    for i, word in enumerate(vocab):
        word = word.decode('utf-8')
        try:
            # again, +4 for the four special tokens in our vocab
            embedding_matrix[i+4] = w2v[word]
            token2word[i+4] = word
            word2token[word] = i+4
        except KeyError as e:
            # skip any oov words from the perspective of our trained w2v model
            continue
    # save the two dicts
    with open(os.path.join(data_dir, 'token2word.json'), 'w') as f:
        json.dump(token2word, f)
    with open(os.path.join(data_dir, 'word2token.json'), 'w') as f:
        json.dump(word2token, f)
    return embedding_matrix

def main(data_dir):
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)

    batch_size = 128
    max_vocab_size = 20000
    max_seq_len = 30
    embedding_dim = 100
    lstm_dim = 128
   

    vectorizer = TextVectorization(max_tokens=max_vocab_size,
                                   output_sequence_length=max_seq_len)
    text_data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
    print('Building vocabulary')
    vectorizer.adapt(text_data)
    vocab = vectorizer.get_vocabulary()
    # load pre-trained w2v model 
    w2v = Word2Vec.load(os.path.join(data_dir, 'processed/w2v.model'))
    print('Building embedding matrix')
    # This matrix will be used to initialze weights in the embedding layer
    embedding_matrix = build_embedding_mat(data_dir, vocab, w2v)
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))

    X_train = vectorizer(np.array([[s] for s in x_train])).numpy()
    X_val = vectorizer(np.array([[s] for s in x_val])).numpy()
    X_test = vectorizer(np.array([[s] for s in x_test])).numpy()
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    acc_scores={}
    dropout=0.7
    layer = "sigmoid"
    print("Building the model with ",layer," and dropout ",dropout)
    model = Sequential()
    model.add(Embedding(input_dim=max_vocab_size+3, output_dim=100, input_length=max_seq_len, weights = [embedding_matrix], trainable=True))
    model.add(Flatten())
    model.add(Dense(lstm_dim,activation=layer
                           , kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
        ))
    model.add(Dropout(dropout))
    model.add(Dense(2,activation='softmax',name='output_layer'))

    print(model.summary())

    print("Compiling the model")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    
    print("Fitting the model")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_val, y_val))
    scores = model.evaluate(X_val, y_val)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    acc_scores[layer+"_val"+str(dropout)] = scores[1]*100
    print("Evaluating model on test data")
    scores = model.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    acc_scores[layer+"_test"+str(dropout)] = scores[1]*100
    # model.save(os.path.join(data_dir, 'processed/'+layer+str(dropout)) )
    model.save(os.path.join(data_dir, 'processed/'+layer+'.model'))
    print(acc_scores)

if __name__ == "__main__":
    main(sys.argv[1])