import os
import sys
from pprint import pprint
import pickle
from gensim.models import Word2Vec

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split('\n')) for line in data]

def main(data_dir):
    pos_lines = read_csv(os.path.join(data_dir, 'pos.txt'))
    neg_lines = read_csv(os.path.join(data_dir, 'neg.txt'))
    all_lines = pos_lines + neg_lines
    all_lines = [line.strip().split() for line in all_lines]
    pprint(all_lines[:10])
    w2v = Word2Vec(all_lines, size=100, window=5, min_count=1, workers=4, sg = 1)
    w2v.save('data/processed/w2v.model')


if __name__ == "__main__":
    main(sys.argv[1])
