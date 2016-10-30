#####################################################################

# Example : decision tree learning
# basic illustrative python script

# For use with test / training datasets : optdigits.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 School of Engineering & Computing Science,
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

reader=csv.reader(open("optdigits.train","rb"),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-63, class label in column 65
        attribute_list.append(list(row[i] for i in (range(0,64))))
        label_list.append(row[64])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("optdigits.test","rb"),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-63, class label in column 65
        attribute_list.append(list(row[i] for i in (range(0,64))))
        label_list.append(row[64])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Decision Tree

# construct parameters as python dictionary

parameters = dict(max_categories = 15,      # max number of categories (use sub-optimal algorithm for larger numbers)
                  max_depth = 25,           # max tree depth
                  min_sample_count = 5,     # min sample count
                  cv_folds = 15,            # theq number of cross-validation folds/iterations
                  use_surrogates = False,   # compute surrogate split, no missing data
                  use_1se_rule = False,     # use 1SE rule => smaller tree
                  truncate_pruned_tree = True, # throw away the pruned tree branches
                  regression_accuracy = 0,  # regression accuracy: N/A here
                  priors = [1,1,1,1,1,1,1,1,1,1]) # the array of priors, the bigger weight, the more attention to the assoc. class
                                            #  (i.e. a case will be judjed to be maligant with bigger chance))


# specify that the types of our attributes is numerical with a categorical output
# and we have 65 of them (64 attributes + 1 class label)

var_types = np.array([cv2.CV_VAR_NUMERICAL] * 64 + [cv2.CV_VAR_CATEGORICAL], np.uint8)

# define and train decision tree object

dtree = cv2.DTree()
dtree.train(training_attributes, cv2.CV_ROW_SAMPLE, training_class_labels, None, None, var_types, None, params=parameters)

############ Perform Testing -- Decision Tree

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform decision tree prediction (i.e. classification)

    result = dtree.predict(testing_attributes[i,:])

    print "Test data example : " + str(i + 1) + " : result = " + str(int(result))

    # record results as either being correct or wrong

    if (result == testing_class_labels[i]) : correct+=1
    elif (result != testing_class_labels[i]) : wrong+=1

# output summmary statistics

total = wrong + correct

print
print "Testing Data Set Performance Summary"
print "Total Correct : "+ str(round((correct / float(total)) * 100, 2)) + "%"
print "Total Wrong : "+ str(round((wrong / float(total)) * 100, 2)) + "%"

#####################################################################
