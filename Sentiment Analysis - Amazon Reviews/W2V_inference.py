  
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
    
# After training our Word2Vec skip-gram model to reconstruct linguistic contexts of words, 
# we find that not all words similar to "good" are positive and same is the case for "bad". 
# This could be because Skip-gram predicts surrounding context words from the target words. 
# So, words like bad might have a positive context like "not bad" - hence similar to good. 
# Similarly, if choosing CBOW, it predicts target words from the surrounding context words. 
# So when different words (semantically different individually) are similar in context, 
# then Word2Vec will have similar outputs when these words are passed as inputs, 
# that is the computed word vectors (in the hidden layer) for these words will be similar.
