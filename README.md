# Hand_Written_Digit_Recognition
Hand-Written-Digit-Recognition
This is a repository with various classifier written in jupyter notebook that works on hand written digits images.The data set can be downloaded from this link. Download the zip file and extract all the images before running ipynb files and also Hand Written Data Recognition Using Logistic Regression.ipynb should be run first so that pickle files are created

Hand Written Data Recognition Using Logistic Regression.ipynb
This is a jupyter notebook that reads images converts them into numpy array using scipy's ndimage and then each digit's pickle file is created pickle files are easy way to store python objects and also they are very fast to work with Now the data can be easily load for other scripts as well.These data are then fed LogisticRegression classifier using sklearn The following observations were noted

Observation
Classifier Properties	LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False, fit_intercept=True, intercept_scaling=1.0, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
Training Time	1182 sec= 20mins
Training Accuracy(accuracy_score)	92
Testing Accuracy(accuracy_score)	92
Testing time	< 1sec
Hand Written Data Recognition Using KNN alogorithm.ipynb
Simple script in which data is used from the generated pickle files and then fed into sklearn's KNNClassifier and the following observations were recorded
Observation
Classifier Properties	KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')
Training Time	18secs
Training Accuracy(accuracy_score)	92.8
Testing Accuracy(accuracy_score)	97.4
Testing time	64sec = 1 mins
Hand Written Data Recognition Using SVC.ipynb
Simple script in which data is used from the generated pickle files and then fed into sklearn's Support Vector Machine Classifier and the following observations were recorded
Observation
Classifier Properties	SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
Training Time	620secs =18mins
Training Accuracy(accuracy_score)	94.864
Testing Accuracy(accuracy_score)	93.1
Testing time	259sec = 4 mins
Hand Written Data Recognition Using GaussianNB.ipynb
Simple script in which data is used from the generated pickle files and then fed into sklearn's Gaussian Naive Bayes and the following observations were recorded
Observation
Classifier Properties	GaussianNB(priors=None)
Training Time	0.50 seconds
Training Accuracy(accuracy_score)	55.212
Testing Accuracy(accuracy_score)	55.54
Testing time	1 sec
Hand Written Data Recognition Using NeuralNetwork.ipynb
Simple script in which data is used from the generated pickle files and then fed into a neural network implemented in tensorflow and the following observations were recorded
Observation
Classifier Properties	total_layers=7 Layer_Units={1:38,2:38,3:38,4:38,5:38,6:38,7:10} Activation functions for layer 1-6 ReLU Activatioin function for output layer Softmax Optimizer used :- Adam Optimizer Learning rate:- 0.001 Steps used :- 1500 Training type:- Full batch
Training Time	2820.50 seconds= 47mins
Training Accuracy(tf.metrics.accuracy)	99.7
Testing Accuracy(tf.metrics.accuracy)	99.2
Testing time	< 1 sec
Hand Written Data Recognition Using ConvolutionNeuralNetwork.ipynb
Simple script in which data is used from the generated pickle files and then fed into a neural network(with CNN) implemented in tensorflow and the following observations were recorded
Observation
Classifier Properties	total_layers=3 Layer_1(Convolutional layer with maxpool) (CNN(kernel=[4,4,1,8],strides=[1,1,1,1],padding='SAME' )->ReLU-> Maxpooling(padding='SAME',ksize=[1,8,8,1],strides=[1,8,8,1])) Layer_2(Convolutional layer with maxpool) (CNN(kernel=[2,2,8,16],strides=[1,1,1,1],padding='SAME' )->ReLU-> Maxpooling(padding='SAME',ksize=[1,4,4,1],strides=[1,4,4,1])) Layer_3(Fully Connected Classifier)(Neurons =10, NO ACTIVATION FUNCTION) Optimizer used :- Adam Optimizer Learning rate:- 0.001 Steps used :- 1500 Training type:- Full batch
Training Time	14646 seconds= 244mins=4hrs
Training Accuracy(tf.metrics.accuracy)	95
Testing Accuracy(tf.metrics.accuracy)	95
Testing time	< 1 sec
Hand Written Data Recognition in NeuralNetwork using Keras.ipynb
Same neural network as Hand Written Data Recognition Using NeuralNetwork.ipynb implemented in Keras
