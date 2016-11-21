#####################################################################

# Example : boosted tree learning
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

number_of_classes = 10

########### unrolling routines

# Acknowledgement: OpenCV 2.4.x example - letter_recog.py

def unroll_samples(number_of_class, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * number_of_class, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, number_of_class, axis=0)
        new_samples[:,-1] = np.tile(np.arange(number_of_class), sample_n)
        return new_samples

def unroll_sample(number_of_class, sample):
        var_n = sample.shape[0]
        new_samples = np.zeros((number_of_class, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat([sample], number_of_class, axis=0)
        new_samples[:,-1] = np.arange(number_of_class)
        return new_samples

def unroll_responses(number_of_class, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*number_of_class, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*number_of_class )
        new_responses[resp_idx] = 1
        return new_responses

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("optdigits.train","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-63, class label in column 65
        attribute_list.append(list(row[i] for i in (range(0,64))))
        label_list.append(row[64])

training_attributes=unroll_samples(number_of_classes, np.array(attribute_list).astype(np.float32))
training_class_labels=unroll_responses(number_of_classes, np.array(label_list).astype(np.float32))

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

############ Perform Training -- Boosted Trees

# define boosted tree object

boostedTree = cv2.ml.Boost_create();

# set parameters (decision tree parameters + those that control boosting)

boostedTree.setCVFolds(0);                # the number of cross-validation folds/iterations - fix at 1
boostedTree.setMaxCategories(10);         # max number of categories (use sub-optimal algorithm for larger numbers)
boostedTree.setMaxDepth(3);               # max tree depth (low -> weak classifiers)
boostedTree.setMinSampleCount(5);         # min sample count
boostedTree.setPriors(np.float32([1,1])); # the array of priors, the bigger weight, the more attention to the assoc. class
                                          # (i.e. a case will be judjed to be maligant with bigger chance))
                                          # N.B. here set as length 2 as we are using 2-class unrolling
boostedTree.setRegressionAccuracy(0);     # regression accuracy: N/A here
boostedTree.setTruncatePrunedTree(True);  # throw away the pruned tree branches
boostedTree.setUse1SERule(False);         # use 1SE rule => smaller tree
boostedTree.setUseSurrogates(False);      # compute surrogate split, no missing data

boostedTree.setBoostType(cv2.ml.BOOST_REAL); # suited to classification problems (see manual for other options)
boostedTree.setWeakCount(100);                 # number of weak classifiers in the boosted ensemble
boostedTree.setWeightTrimRate(0.95);         # threshold in range {0->1} to save computation. Training examples with weight < (1 - trim_rate) as excluded from next training iteration

# specify that the types of our variables is numerical and we have 65 + 1 of them (64 attributes + 1 class label + 1 unrolled label)
# class labels are categorical as {0,1} (and must be set as so for the OpenCV implmentation to work)

var_types = np.array([cv2.ml.VAR_NUMERICAL] * 64 + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL], np.uint8)

# train boosted tree object

boostedTree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

############ Perform Testing -- Boosted Trees

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform boosted tree prediction (i.e. classification)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        # As currently the boosted tree classifier in OpenCV can only be trained
        # for 2-class problems, we transform the training data set by
        # "unrolling" each training sample as many times as the number of
        # classes (10) that we have.
        #
        #  In "unrolling" we add an additional attribute to each training
        #  sample that contains the classification - here 10 new samples
        #  are added for every original sample, one for each possible class
        #  but only one with the correct class as an additional attribute
        #  value has a new binary class of 1, all the rest of the new samples
        #  have a new binary class of 0.
        #
        #  The boosted tree classifier is then trained and tested on this
        #  unrolled data set.
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    unrolled_testing_examples = unroll_sample(number_of_classes, testing_attributes[i,:])

    # we perform testing over all the unrolled examples and take the maximum result
    # following the example at:
    # https://github.com/opencv/opencv/blob/master/samples/python/letter_recog.py

    # result should contain the number of trees that voted for each unrolled
    # example, hence each class

    _, result = boostedTree.predict(unrolled_testing_examples);

    result = result.reshape(-1, number_of_classes).argmax(1);

    print("Test data example : {} : result =  {}".format((i+1), int(result)));
    # record results as either being correct or wrong

    if (result == testing_class_labels[i]) : correct+=1
    elif (result != testing_class_labels[i]) : wrong+=1

# output summmary statistics

total = wrong + correct

print();
print("Testing Data Set Performance Summary");
print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

#####################################################################
