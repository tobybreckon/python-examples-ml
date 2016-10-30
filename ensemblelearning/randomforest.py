#####################################################################

# Example : decision (random) forest learning
# basic illustrative python script

# For use with test / training datasets : optdigits.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Version : with OpenCV 3 / Python 3 fixes

# Copyright (c) 2014 / 2016 School of Engineering & Computing Sciences,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

### classes as integers {0 ... 9}

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("optdigits.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-63, class label in column 65
        attribute_list.append(list(row[i] for i in (range(0,64))))
        label_list.append(row[64])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("optdigits.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-63, class label in column 65
        attribute_list.append(list(row[i] for i in (range(0,64))))
        label_list.append(row[64])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Random Forest

# define random forest object

dforest = cv2.ml.RTrees_create();

# set parameters (decision tree parameters + those that control forest)

dforest.setCVFolds(1);                # the number of cross-validation folds/iterations - fix at 1
dforest.setMaxCategories(5);         # max number of categories (use sub-optimal algorithm for larger numbers)
dforest.setMaxDepth(5);               # max tree depth (low -> weak classifiers)
dforest.setMinSampleCount(5);         # min sample count
dforest.setPriors(np.float32([1,1,1,1,1,1,1,1,1,1])); # the array of priors, the bigger weight, the more attention to the assoc. class
                                          # (i.e. a case will be judjed to be maligant with bigger chance))
                                          # N.B. here set as length 2 as we are using 2-class unrolling
dforest.setRegressionAccuracy(0);     # regression accuracy: N/A here
dforest.setTruncatePrunedTree(True);  # throw away the pruned tree branches
dforest.setUse1SERule(False);         # use 1SE rule => smaller tree
dforest.setUseSurrogates(False);      # compute surrogate split, no missing data

dforest.setActiveVarCount(5);         # number of randomly selected subset of attributes used at each tree node to find the best split(s).

max_num_of_trees_in_the_forest = 3000;  # typically the more trees you have the better the accuracy.
forest_accuracy = 0.001;                # sufficient accuracy on task
dforest.setTermCriteria((cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, max_num_of_trees_in_the_forest, forest_accuracy));

# specify that the types of our attributes is numerical with a categorical output
# and we have 65 of them (64 attributes + 1 class label)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 64 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

# define and train random forest object

dforest.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

############ Perform Testing -- Decision Forest

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    _, result = dforest.predict(testing_attributes[i,:], cv2.ml.ROW_SAMPLE);

    # and for undocumented reasons take the first element of the resulting array
    # as the result

    print("Test data example : {} : result =  {}".format((i+1), int(result[0])));

    # record results as either being correct or wrong

    if (result[0] == testing_class_labels[i]) : correct+=1
    elif (result[0] != testing_class_labels[i]) : wrong+=1

# output summmary statistics

total = wrong + correct

print();
print("Testing Data Set Performance Summary");
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

#####################################################################
