  
import sys
from gensim.models import Word2Vec


def main(text_path):
    with open(text_path) as f:
        sample_text = f.readlines()

    sample_text = [w.strip() for w in sample_text]
    
    w2v = Word2Vec.load('data/processed/w2v.model')

    # print(w2v.most_similar('good')[:5])
    
    return ['{} => \n {}'.format(w, [o[0] for o in w2v.most_similar([w2v[w]],topn = 21)[1:]]) for w in sample_text]


if __name__ == '__main__':
    most_similar = main(sys.argv[1])
    print('\n\n'.join(most_similar))