#####################################################################

# Example : decision tree learning
# basic illustrative python script

# For use with test / training datasets : optdigits.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
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

############ Perform Training -- Decision Tree

# define decision tree object

dtree = cv2.ml.DTrees_create();

# set parameters (changing may or may not change results)

dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fix at 1
dtree.setMaxCategories(15); # max number of categories (use sub-optimal algorithm for larger numbers)
dtree.setMaxDepth(25);       # max tree depth
dtree.setMinSampleCount(5); # min sample count
dtree.setPriors(np.float32([1,1,1,1,1,1,1,1,1,1]));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                     #  (i.e. a case will be judjed to be maligant with bigger chance))
dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
dtree.setUse1SERule(False);      # use 1SE rule => smaller tree
dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

# specify that the types of our attributes is numerical with a categorical output
# and we have 65 of them (64 attributes + 1 class label)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 64 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

# train decision tree object

dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

############ Perform Testing -- Decision Tree

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    _, result = dtree.predict(testing_attributes[i,:],  cv2.ml.ROW_SAMPLE);

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
