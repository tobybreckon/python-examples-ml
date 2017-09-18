#####################################################################

# Example : Support Vector Machine (SVM) learning
# basic illustrative python script - fixed kernel, use of

# For use with test / training datasets : semeion.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Version : with OpenCV 3 / Python 3 fixes

# Copyright (c) 2016 School of Engineering & Computing Sciences,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

### classes as integers {0 ... 9}

## use SVM auto-training (grid search)
# if available in python bindings; see open issue: https://github.com/opencv/opencv/issues/7224

use_svm_autotrain = False;

########### Load Training and Testing Data Sets

# load training data set (N.B. delimiter is space, not comma)

reader=csv.reader(open("semeion.train","rt", encoding='ascii'),delimiter=' ')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-255, class label in column 256
        attribute_list.append(list(row[i] for i in (range(0,256))))
        label_list.append(row[256])

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.integer)

# load testing data set (N.B. delimiter is space, not comma)

reader=csv.reader(open("semeion.test","rt", encoding='ascii'),delimiter=' ')

attribute_list = []
label_list = []

for row in reader:
        # attributes in columns 0-255, class label in column 256
        attribute_list.append(list(row[i] for i in (range(0,256))))
        label_list.append(row[256])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.integer)

############ Perform Training -- SVM

# define SVM object

svm = cv2.ml.SVM_create();

# set kernel
# choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

svm.setKernel(cv2.ml.SVM_LINEAR);

# set parameters (some specific to certain kernels)

svm.setC(1.0); # penalty constant on margin optimization
svm.setType(cv2.ml.SVM_C_SVC); # multiple class (2 or more) classification
svm.setGamma(0.5); # used for SVM_RBF kernel only, otherwise has no effect
svm.setDegree(3);  # used for SVM_POLY kernel only, otherwise has no effect

# set the relative weights importance of each class for use with penalty term

svm.setClassWeights(np.float32([1,1,1,1,1,1,1,1,1,1]));

# define and train svm object

if (use_svm_autotrain) :

    # use automatic grid search across the parameter space of kernel specified above
    # (ignoring kernel parameters set previously)

    # if it is available : see https://github.com/opencv/opencv/issues/7224

    svm.trainAuto(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int)), kFold=10);
else :

    # use kernel specified above with kernel parameters set previously

    svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels);

############ Perform Testing -- SVM

correct = 0 # handwritten digit correctly identified
wrong = 0   # handwritten digit wrongly identified

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # (to get around some kind of OpenCV python interface bug, vertically stack the
    #  example with a second row of zeros of the same size and type which is ignored).

    sample = np.vstack((testing_attributes[i,:],
                        np.zeros(len(testing_attributes[i,:])).astype(np.float32)));

    # perform SVM prediction (i.e. classification)

    _, result = svm.predict(sample, cv2.ml.ROW_SAMPLE);

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
