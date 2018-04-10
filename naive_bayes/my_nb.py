#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import pickle
import os
from sklearn import cross_validation

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectPercentile, f_classif


path = "C:\\Users\\user\\Documents\\GitHub\\my_ud120\\my_ud120\\naive_bayes"
os.chdir(path)
sys.path.append("../tools")

#########################################################
### word_data to unix ###

original = "../tools/word_data.pkl"
destination = "../tools/word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))

### preprocess ###

pkl_file = open('../tools/email_authors.pkl', 'rb')
authors = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../tools/word_data_unix.pkl', 'rb')
word_data = pickle.load(pkl_file)
pkl_file.close()

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)

features_train = selector.transform(features_train_transformed).toarray()
features_test = selector.transform(features_test_transformed).toarray()

### info on the data
print("no. of Chris training emails:", sum(labels_train))
print("no. of Sara training emails:", len(labels_train)-sum(labels_train))

#########################################################
### your code goes here ###


from sklearn.naive_bayes import GaussianNB

### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")


### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print("training time:", round(time()-t0, 3), "s")


### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
accuracy = sum(labels_test == pred)/pred.shape

#########################################################
















