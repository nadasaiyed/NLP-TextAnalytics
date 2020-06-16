The models were tuned on various hyperparameters like 
alpha values (0.01,0.5,1), min_df (0.0001,0.001,default=1) and max_df(0.1,0.5,default=1).
It was found that the best validation accuracy was obtained for default values for min_df and max_df and alpha value = 0.5.
Based of these hyperparameters the questions were asnwered below and the model for uni/bi-grams were tested.
The validation and test scores are given as the output.
The best performance is given by classifier which uses unigram+bigram with accuracy of 83.49% on test data set 
and 83.6% on validation dataset

1.
    Amongst both the conditions, With stopwords performed slightly better with smoothining parameter as 0.5,
    giving accuracy around 83% for ngrams as unigram+bigram. This could be because the stopwords  add significant information,
    hence not removing them helped in extracting the actual features of the document which contribute towards
    the sentiment of the document. This is specially for cases where a positive word is combined with a negation.
    For instance, 'i didn't like the product' after stopword removal becomes 'like product' which does not predict the actual polarity.
    
2. 
    unigrams+bigrams condition performs better than just using unigrams as our feature or just using bigram. 
    This difference can be attributed to the fact that for detecting the polarity of a review, 
    there could be positive words which could be clubbed with negative words like 'not happy'. 
    For such bigrams instead of classifying it as positive because of the occurance of word 'happy',
    its more accurate to classify it as negative review using the bigram. However some bigrams might belittle
    the effects of unigram words (adjectives) which might be important features. Hence, using unigram+bigram performs the best.
    
    Stopwords removed 	text features	Accuracy (test set)
    Yes             	Uni	            0.8039375
    Yes	                Bi	            0.7827875
    Yes	                Uni + Bi	    0.821425
    No	                Uni	            0.809325
    No	                Bi          	0.8244875
    No              	Uni + Bi    	0.834925

