#####################################################################
# Example : k-Means clustering
# basic illustrative python script

# For use with test / training datasets : spambase.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 School of Engineering & Computing Sciences,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

# N.B. we are also relyng on matplotlib and scipy being installed
# for this example

from matplotlib import pyplot as plt
from scipy import stats

########### Define classes

classes = {1 : 'spam', 0  : 'ham'} # "ham" = mail that is not spam
inv_classes = {v: k for k, v in list(classes.items())}

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("spambase.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (list(range(0,57)))))
        label_list.append(row[57])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("spambase.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (list(range(0,57)))))
        label_list.append(row[57])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Clustering -- k-Means

# define termination criteria and apply kmeans() with k = 2

# N.B. in this example we are assuming two clusters - one for each class.
# This assumption may not be valid

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001);

# Set flags to initialize with random centres

flags = cv2.KMEANS_RANDOM_CENTERS;

# perform k-means clustering

compactness,labels,centers = cv2.kmeans(training_attributes,2,None,criteria,10, flags)

# take the most common class label for each cluster to work out which is which
# (using statistical mode)

ModeC0,number = stats.mode(training_class_labels[labels.ravel()==0])
ModeC1,number = stats.mode(training_class_labels[labels.ravel()==1])

# get the corresponding cluster centres the right way around

if (ModeC0 == inv_classes['spam']): # if cluster C0 is mainly spam examples
    spam  = centers[0,:]
    ham = centers[1,:]
else: # else cluster C1 is mainly spam examples
    spam  = centers[1,:]
    ham = centers[0,:]

############ Perform Testing -- k-Means nearest cluster

tp = 0 # spam
tn = 0 # ham
fp = 0 # classed as spam, but is ham
fn = 0 # classed as ham, but is spam

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # very simple cluster assignment test - find nearest cluster centre based
    # on Euclidean distance to the "spam" cluster centre or the "ham" cluster centre

    # N.B. a better way would be to calculate the mean/co-variance of each cluster
    # and use use the malanobis distance or perhaps another distance measure

    result = np.argmin((np.linalg.norm(ham-testing_attributes[i,:]), np.linalg.norm(spam-testing_attributes[i,:])))

    print("Test data example : " + str(i + 1) + " : result = " + str(classes[int(result)]))

    # record results as tp/tn/fp/fn

    if (result == testing_class_labels[i] == 1) : tp+=1
    elif (result == testing_class_labels[i] == 0) : tn+=1
    elif (result != testing_class_labels[i]) :
        if ((result == 1) and (testing_class_labels[i] == 0)) : fp+=1
        elif ((result == 0) and (testing_class_labels[i] == 1)) : fn+=1

# output summmary statistics

total = tp + tn + fp + fn
correct = tp + tn
wrong = fp + fn

print()
print("Testing Data Set Performance Summary")
print("TP : " + str(round((tp / float(total)) * 100, 2)) + "%")
print("TN : " + str(round((tn / float(total)) * 100, 2)) + "%")
print("FP : " + str(round((fp / float(total)) * 100, 2)) + "%")
print("FN : " + str(round((fn / float(total)) * 100, 2)) + "%")
print("Total Correct : "+ str(round((correct / float(total)) * 100, 2)) + "%")
print("Total Wrong : "+ str(round((wrong / float(total)) * 100, 2)) + "%")

#####################################################################
