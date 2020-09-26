from sklearn import model_selection, naive_bayes, metrics
from sklearn.feature_extraction.text import  CountVectorizer
import csv
import sys

""" This is a Naive Bayes Classifier to classify whether a chunk of data describes an ad for an office
    space available for rent"""

    
chunks = []
has_space = []

#read data from file and store in chunks and has_space lists
with open('ra_data_classifier.csv', 'r', encoding='mac_roman') as csvfile:
    fileReader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for row in fileReader:
        chunks.append(row[1])
        has_space.append(row[2])


#split data into training and testing 
train_chunks, test_chunks, train_has_space, test_has_space = model_selection.train_test_split(chunks, has_space, test_size = 0.25)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(chunks)


# transform the training and validation data using count vectorizer object
train_chunks_count =  count_vect.transform(train_chunks)
test_chunks_count =  count_vect.transform(test_chunks)

#fit the training data using a Naive Bayes Classifier
best_fit = naive_bayes.MultinomialNB().fit(train_chunks_count, train_has_space)
    
# predict the labels on validation dataset
predictions = best_fit.predict(test_chunks_count)

#get the accuracy of the classifier
accuracy = metrics.accuracy_score(predictions, test_has_space)

print ("Accuracy of Naive Bayes classifier is %f" %accuracy)

""" This part of the code can be used to predict whether a new input chunk classifies as an office space
    available for rent or not. The new input is given by the user """

#This additional code is commented. Depending on the needs of the user, it can be used

"""
input_chunk=sys.argv[1]
input_chunk_list = []
input_chunk_list.append(input_chunk)
imput_chunk_count = count_vect.transform(input_chunk_list)
predict_input = best_fit.predict(imput_chunk_count)
print('Prediction for Input chunk : '+str(input_chunk)+' - '+str(predict_input))
"""
