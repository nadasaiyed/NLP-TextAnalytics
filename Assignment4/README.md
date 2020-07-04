Dropout	Sigmoid	ReLu	Tanh

0.3	79.339	79.03125	78.80375

0.5	79.431	78.8025	79.01875

0.7	79.735	79.3637	78.83375


Considering 0.7 as the best dropout rate for hidden layer, we add l2 norm regularization for sigmoid and ReLu and for tanh we proceed with droupout rate as 0.5

An activation function takes into account the interaction effects in different parameters and does a transformation after which it gets to decide which neuron passes forward the value into the next layer. We see that sigmoid performs the best, this could be due to the fact that ReLu ranges from 0 to infinity and negative values are converted to zero. However, for classification, its better to get values between 0 and 1 to predict the probability of the class. Similarly, for tanh, it being symmetric around the origin (due to gradient being steeper than sigmoid) gives us values in the range of -1 to 1, which could slightly decrease the accuracy.

Dropout rate is a technique to randomly remove neurons so as to avoid overfitting. It is common for larger networks (more layers or more nodes) to more easily overfit the training data. Ideal dropout rate for hidden layer suggested is 0.5-0.8, hence we can see that accuracy with 0.7 is the best. A dropout with 0.7 effectively allows us to train and sample from a probability distribution of netwrok architectures. Dropping a neuron with 0.7 probability gets the highest variance for this distribution.

After adding the l2 norm regularization (0.001), the accuracy for sigmoid increase to 81.127%, for relu 80.33% and for tanh 80.81. Regularization is also a similar technique to minimize the generalization error and prevent the model from becoming overfit. Thus, we see that we get a slight lower accuracy on train and val datasets as compared to test data sets after adding this regularization, which could be due to regularization preventing the model to learn more and more about the training dataset.

Thus, by combining the best dropout rates and adding l2 norm regularization, we get best accuracy for sigmoid 

