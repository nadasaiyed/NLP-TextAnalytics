import os
import sys
from pprint import pprint
import sklearn
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_test = labels[len(x_train): len(x_train)+len(x_val)]
    y_val = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test


def train(x_train, y_train,ngram,stopword,data_dir,pkl_name):

    count_vect = CountVectorizer(ngram_range=ngram,stop_words=stopword)
    x_train_count = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    clf = MultinomialNB(alpha=0.5).fit(x_train_tfidf, y_train)
    with open(os.path.join(data_dir, pkl_name),'wb') as f:
        pickle.dump(clf,f)
    return clf, count_vect, tfidf_transformer
   
def mnb_uni(x_train,y_train,data_dir,alpha):
    print("Training the model for unigrams with stopwords")
    return train(x_train,y_train,(1,1),None,data_dir,'mnb_uni.pkl',alpha)

def mnb_bi(x_train,y_train,data_dir):
    print("Training the model for bigrams with stopwords")
    return train(x_train,y_train,(2,2),None,data_dir,'mnb_bi.pkl')

def mnb_uni_bi(x_train,y_train,data_dir):
    print("Training the model for unigrams-bigrams with stopwords")
    return train(x_train,y_train,(1,2),None,data_dir,'mnb_uni_bi.pkl')

def mnb_uni_ns(x_train,y_train,data_dir):
    print("Training the model for unigrams without stopwords")
    return train(x_train,y_train,(1,1),'english',data_dir,'mnb_uni_ns.pkl')
    
def mnb_bi_ns(x_train,y_train,data_dir):
    print("Training the model for bigrams without stopwords")
    return train(x_train,y_train,(2,2),'english',data_dir,'mnb_bi_ns.pkl')
    
def mnb_uni_bi_ns(x_train,y_train,data_dir):
    print("Training the model for unigrams-bigrams without stopwords")
    return train(x_train,y_train,(1,2),'english',data_dir,'mnb_uni_bi_ns.pkl')

def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        }

def main(data_dir):
    
    print(data_dir)
    # load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)
    scores = {}

    # best_alpha=1
    # best_acc=0
    # for alpha in ([0.01,0.5,0.75]) :
    #     #Train data for mnb_uni
    #     clf, count_vect, tfidf_transformer = mnb_uni_bi(x_train, y_train,data_dir,alpha)
    #     # validate for mnb_uni
    #     print('Validating for alpha', alpha)
    #     scores['mnb_uni_val'+'_'+str(alpha)] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    #     if(scores['mnb_uni_val'+'_'+str(alpha)]['accuracy']>best_acc):
    #         best_acc=scores['mnb_uni_val'+'_'+str(alpha)]['accuracy']
    #         best_alpha=alpha

    #Train data for mnb_uni
    clf, count_vect, tfidf_transformer = mnb_bi(x_train, y_train,data_dir)
    # validate for mnb_uni
    print('Validating')
    scores['mnb_uni_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_uni
    print('Testing')
    scores['mnb_uni_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    #Train data for mnb_bi
    clf, count_vect, tfidf_transformer = mnb_bi(x_train, y_train,data_dir)
    # validate for mnb_bi
    print('Validating')
    scores['mnb_bi_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_bi
    print('Testing')
    scores['mnb_bi_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    #Train data for mnb_uni_bi
    clf, count_vect, tfidf_transformer = mnb_uni_bi(x_train, y_train,data_dir)
    # validate for mnb_uni_bi
    print('Validating')
    scores['mnb_uni_bi_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_uni_bi
    print('Testing')
    scores['mnb_uni_bi_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    #Train data for mnb_uni_ns
    clf, count_vect, tfidf_transformer = mnb_uni_ns(x_train, y_train,data_dir)
    # validate for mnb_uni_ns
    print('Validating')
    scores['mnb_uni_ns_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_uni_ns
    print('Testing')
    scores['mnb_uni_ns_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    #Train data for mnb_bi_ns
    clf, count_vect, tfidf_transformer = mnb_bi_ns(x_train, y_train,data_dir)
    # validate for mnb_bi_ns
    print('Validating')
    scores['mnb_bi_ns_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_bi_ns
    print('Testing')
    scores['mnb_bi_ns_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    #Train data for mnb_uni_bi_ns
    clf, count_vect, tfidf_transformer = mnb_uni_bi_ns(x_train, y_train,data_dir)
    # validate for mnb_uni_bi_ns
    print('Validating')
    scores['mnb_uni_bi_ns_val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test for mnb_uni_bi_ns
    print('Testing')
    scores['mnb_uni_bi_ns_test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    return scores    


if __name__ == '__main__':
    pprint(main(sys.argv[1]))
