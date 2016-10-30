#####################################################################

# Example : decision tree learning
# basic illustrative python script

# For use with test / training datasets : wdbc.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014  / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html
# Version : 0.3 (OpenCV 3 / Python 3 fixes)

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

classes = {'M': 1, 'B': 0}
inv_classes = {v: k for k, v in classes.items()}

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("wdbc.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 2-32, class label in column 1,
        # ignore patient ID in column 0
        attribute_list.append(list(row[i] for i in (range(2,32))))
        label_list.append(classes[row[1]])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("wdbc.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 2-32, class label in column 1,
        # ignore patient ID in column 0
        attribute_list.append(list(row[i] for i in (range(2,32))))
        label_list.append(classes[row[1]])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Decision Tree

# define decision tree object

dtree = cv2.ml.DTrees_create();

# set parameters (changing may or may not change results)

dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fixed at 1
dtree.setMaxCategories(15); # max number of categories (use sub-optimal algorithm for larger numbers)
dtree.setMaxDepth(8);       # max tree depth
dtree.setMinSampleCount(5); # min sample count
dtree.setPriors(np.float32([1,1]));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                     #  (i.e. a case will be judjed to be maligant with bigger chance))
dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
dtree.setUse1SERule(True);      # use 1SE rule => smaller tree
dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

# specify that the types of our attributes is numerical with a categorical output
# and we have 31 of them (30 attributes + 1 class label)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 30 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

# train decision tree object

print("ready");

dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

############ Perform Testing -- Decision Tree

tp = 0 # M
tn = 0 # B
fp = 0 # classed as M, but is B
fn = 0 # classed as B, but is M

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    _, result = dtree.predict(testing_attributes[i,:])

    print("Test data example : {} : result =  {}".format((i+1), inv_classes[int(result[0])]));

    # record results as tp/tn/fp/fn

    if (result[0] == testing_class_labels[i] == 1) : tp+=1
    elif (result[0] == testing_class_labels[i] == 0) : tn+=1
    elif (result[0] != testing_class_labels[i]) :
        if ((result[0] == 1) and (testing_class_labels[i] == 0)) : fp+=1
        elif ((result[0] == 0) and (testing_class_labels[i] == 1)) : fn+=1

# output summmary statistics

total = tp + tn + fp + fn
correct = tp + tn
wrong = fp + fn

print();
print("Testing Data Set Performance Summary");
print("TP : {}%".format(str(round((tp / float(total)) * 100, 2))));
print("TN : {}%".format(str(round((tn / float(total)) * 100, 2))));
print("FP : {}%".format(str(round((fp / float(total)) * 100, 2))));
print("FN : {}%".format(str(round((fn / float(total)) * 100, 2))));
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

#####################################################################
