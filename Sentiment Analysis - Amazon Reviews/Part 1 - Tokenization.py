import random

# Reading the files

f = open("neg.txt", "r")
neg_list=f.read().splitlines()
f = open("pos.txt", "r")
pos_list=f.read().splitlines()
# print(len(neg_list)," ",len(pos_list))
reviews = neg_list+pos_list
# len(reviews)

## Functions to tokenize, normalize, write to csv

def my_tokenizer(sentence):
    return str.split(sentence)

def my_normalizer(sentence,characters):  
    for i in range(len(sentence)):
        for c in characters:
            try:
                sentence[i]=sentence[i].replace(c,"")
                if(len(sentence[i])==0):
                    break
#                 if((len(sentence[i])==1):
#                     sentence[i]=sentence[i].replace('.',"")
#                     sentence[i]=sentence[i].replace('-',"")
#                     sentence[i]=sentence[i].replace(',',"")
#                     break
                elif(sentence[i][-1] in '.,-'):
                    sentence[i]=sentence[i][0:len(sentence[i])-1]    
            except:
                print(sentence)
    return [x for x in sentence if x]

import nltk
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
# stopwords

def remove_stopwords(sent,stopwords):
    for item in stopwords:
        if item in sent:
            sent.remove(item)
    return sent

def split(length,percent):
    count=[]
    count.append(int((length*(percent[0]))-1))
    count.append(int((length*(percent[0]+percent[1]))))
    return count

def write_csv(name,my_list):
    with open(name,'w') as f:
        for sublist in my_list:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    
characters='!"#$%&()*+/:;<=>@[\\]^`{|}~'

## files for words with stopwords

reviews_tokenized = []
reviews_tokenized_no_sw = []
for sent,i in zip(reviews,range(len(reviews))):
    temp = my_tokenizer(sent)
    temp = my_normalizer(temp,characters)
    reviews_tokenized.append(temp)
write_csv('out1.csv',reviews_tokenized)
shuffled=random.sample(reviews_tokenized,len(reviews_tokenized))
s=split(len(shuffled),[0.8,0.1,0.1])
write_csv('train_w_sw.csv',shuffled[0:s[0]])
write_csv('test_w_sw.csv',shuffled[s[0]:s[1]])
write_csv('val_w_sw.csv',shuffled[s[1]:len(shuffled)])


##Files for without stopwords

for sublist in reviews_tokenized:
    remove_stopwords(sublist,stopwords)
write_csv('out2.csv',reviews_tokenized)
random.shuffle(reviews_tokenized)
s=split(len(reviews_tokenized),[0.8,0.1,0.1])
write_csv('train_wo_sw.csv',reviews_tokenized[0:s[0]])
write_csv('test_wo_sw.csv',reviews_tokenized[s[0]:s[1]])
write_csv('val_wo_sw.csv',reviews_tokenized[s[1]:len(reviews_tokenized)])
