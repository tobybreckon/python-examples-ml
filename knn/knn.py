#####################################################################

# Example : k-NN learning
# basic illustrative python script

# For use with test / training datasets : spambase.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

classes = {1 : 'spam', 0  : 'ham (not spam)'}

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("spambase.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (range(0,57))))
        label_list.append(row[57])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("spambase.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (range(0,57))))
        label_list.append(row[57])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- k-NN

# define kNN object

knn = cv2.ml.KNearest_create();

# set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
# on this data set (may not for others - http://code.opencv.org/issues/2661)

knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE);

# set up classification, turning off regression

knn.setIsClassifier(True);

# perform training of k-NN

knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels);

############ Perform Testing -- k-NN

tp = 0 # spam
tn = 0 # ham
fp = 0 # classed as spam, but is ham
fn = 0 # classed as ham, but is spam

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform k-NN prediction (i.e. classification)

    # (to get around some kind of OpenCV python interface bug, vertically stack the
    #  example with a second row of zeros of the same size and type which is ignored).

    sample = np.vstack((testing_attributes[i,:],
                        np.zeros(len(testing_attributes[i,:])).astype(np.float32)))

    # now do the prediction returning the result, results (ignored) and also the responses
    # + distances of each of the k nearest neighbours
    # N.B. k at classification time must be < maxK from earlier training

    _, results, neigh_respones, distances = knn.findNearest(sample, k = 3);

    # print "Test data example : " + str(i + 1) + " : result = " + str(classes[int(result[0])])
    print("Test data example : {} : result =  {}".format((i+1), classes[(int(results[0]))]));

    # record results as tp/tn/fp/fn

    if (results[0] == testing_class_labels[i] == 1) : tp+=1
    elif (results[0] == testing_class_labels[i] == 0) : tn+=1
    elif (results[0] != testing_class_labels[i]) :
        if ((results[0] == 1) and (testing_class_labels[i] == 0)) : fp+=1
    elif ((results[0] == 0) and (testing_class_labels[i] == 1)) : fn+=1

# output summmary statistics

total = tp + tn + fp + fn
correct = tp + tn
wrong = fp + fn

print();
print("Testing Data Set Performance Summary");
print("TP : {}%".format(round((tp / float(total)) * 100, 2)));
print("TN : {}%".format(round((tn / float(total)) * 100, 2)));
print("FP : {}%".format(round((fp / float(total)) * 100, 2)));
print("FN : {}%".format(round((fn / float(total)) * 100, 2)));
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

#####################################################################
