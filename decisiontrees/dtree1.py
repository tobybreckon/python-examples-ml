#####################################################################

# Example : decision tree learning
# basic illustrative python script

# For use with test / training datasets : car.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Version : 0.4 (OpenCV 3 / Python 3 fixes)

# Copyright (c) 2014 /2016 School of Engineering & Computing Sciences,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

classes = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
inv_classes = {v: k for k, v in classes.items()}

########### Define attributes (map to numerical)

attributes = {'vhigh' : 0, 'high' : 1, 'med' : 2,
              'low': 2, '2' : 3, '3': 4, '4': 5,
              '5more': 5, 'more': 6, 'small': 7, 'big': 8}

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("car.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-5, class label in column 6
        attribute_list.append(list(attributes[row[i]] for i in (range(0,6))))
        label_list.append(classes[row[6]])


training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("car.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 1-5, class label in column 6
        attribute_list.append(list(attributes[row[i]] for i in (range(0,6))))
        label_list.append(classes[row[6]])


testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Decision Tree

# define decision tree object

dtree = cv2.ml.DTrees_create();

# set parameters (changing may or may not change results)

dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fix at 1
dtree.setMaxCategories(25); # max number of categories (use sub-optimal algorithm for larger numbers)
dtree.setMaxDepth(8);       # max tree depth
dtree.setMinSampleCount(25); # min sample count
dtree.setPriors(np.float32([1,1,1,1]));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                     #  (i.e. a case will be judjed to be maligant with bigger chance))
dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
dtree.setUse1SERule(True);      # use 1SE rule => smaller tree
dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

# specify that the types of our attributes is ordered with a categorical class output
# and we have 7 of them (6 attributes + 1 class label)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 6 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

# train decision tree object

dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

############ Perform Testing -- Decision Tree

correct = 0
wrong = 0

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    _, result = dtree.predict(testing_attributes[i,:], cv2.ml.ROW_SAMPLE);

    # and for undocumented reasons take the first element of the resulting array
    # as the result

    print("Test data example : {} : result =  {}".format((i+1), inv_classes[int(result[0])]));

    # record results as tp/tn/fp/fn

    if (result[0] == testing_class_labels[i]) : correct+=1
    elif (result[0] != testing_class_labels[i]) : wrong+=1

# output summmary statistics

total = correct + wrong

print();
print("Testing Data Set Performance Summary");
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));

#####################################################################
