#####################################################################

# Example : Neural Network based learning
# basic illustrative python script

# For use with test / training datasets : spambase.{train | test}

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 16 School of Engineering & Computing Sciences,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

########### Define classes

classes = {1 : 'spam', 0  : 'ham (not spam)'}

#####################################################################

########### construct output layer

# expand training responses defined as class labels {0,1...,N} to output layer
# responses for the OpenCV MLP (Neural Network) implementation such that class
# label c becomes {0,0,0, ... 1, ...0} where the c-th entry is the only non-zero
# entry (equal to "value", conventionally = 1) in the N-length vector

# labels : a row vector of class label transformed to {0,0,0, ... 1, ...0}
# max_classes : maximum class label
# value: value use to label the class response in the output layer vector
# sigmoid : {true | false} - return {-value,....value,....-value} instead for
#           optimal use with OpenCV sigmoid function

def class_label_to_nn_output(label, max_classes, is_sigmoid, value):
    if (is_sigmoid):
        output = np.ones(max_classes).astype(np.float32) * (-1 * value)
        output[int(label)] = value
    else:
        output = np.zeros(max_classes).astype(np.float32)
        output[int(label)] = value

    return output

#####################################################################

########### Load Training and Testing Data Sets

# load training data set

reader=csv.reader(open("spambase.train","rt", encoding='ascii'),delimiter=',')


attribute_list = []
label_list = []
nn_outputs_list = []

#### N.B there is a change in the loader here (compared to other examples)

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (list(range(0,57)))))
        label_list.append(row[57])
        nn_outputs_list.append(class_label_to_nn_output(row[57], len(classes), True, 1))

training_attributes=np.array(attribute_list).astype(np.float32)
training_class_labels=np.array(label_list).astype(np.float32)
training_nn_outputs=np.array(nn_outputs_list).astype(np.float32)

# load testing data set

reader=csv.reader(open("spambase.test","rt", encoding='ascii'),delimiter=',')

attribute_list = []
label_list = []
nn_outputs_list = []

for row in reader:
        # attributes in columns 0-56, class label in last column,
        attribute_list.append(list(row[i] for i in (list(range(0,57)))))
        label_list.append(row[57])

testing_attributes=np.array(attribute_list).astype(np.float32)
testing_class_labels=np.array(label_list).astype(np.float32)

############ Perform Training -- Neural Network

# create the network object

nnetwork = cv2.ml.ANN_MLP_create();

# define number of layers, sizes of layers and train neural network
# neural networks only support numerical inputs (convert any categorical inputs)

# set the network to be 2 layer 57->10->2
# - one input node per attribute in a sample
# - 10 hidden nodes
# - one output node per class
# defined by the column vector layer_sizes

layer_sizes = np.int32([57, 10, len(classes)]); # format = [inputs, hidden layer n ..., output]
nnetwork.setLayerSizes(layer_sizes);

# create the network using a sigmoid function with alpha and beta
# parameters = 1 specified respectively (standard sigmoid)

nnetwork.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1, 1);

# available activation functions = (cv2.ml.ANN_MLP_SIGMOID_SYM or cv2.ml.ANN_MLP_IDENTITY, cv2.ml.ANN_MLP_GAUSSIAN)

# specify stopping criteria and backpropogation for training

nnetwork.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP);
nnetwork.setBackpropMomentumScale(0.1);
nnetwork.setBackpropWeightScale(0.1);

nnetwork.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 1000, 0.001))

        ## N.B. The OpenCV neural network (MLP) implementation does not
        ## support categorical variable output explicitly unlike the
        ## other OpenCV ML classes.
        ## Instead, following the traditional approach of neural networks,
        ## the output class label is formed by we a binary vector that
        ## corresponds the desired output layer result for a given class
        ## e.g. {0, 0 ... 1, 0, 0} components (one element by class) where
        ## an entry "1" in the i-th vector position correspondes to a class
        ## label for class i
        ## for optimal performance with the OpenCV intepretation of the sigmoid
        ## we use {-1, -1 ... 1, -1, -1}

        ## prior to training we must construct these output layer responses
        ## from our conventional class labels (carried out by class_label_to_nn_output()

# train the neural network (using training data)

nnetwork.train(training_attributes, cv2.ml.ROW_SAMPLE, training_nn_outputs);

############ Perform Testing -- Neural Network

tp = 0 # spam
tn = 0 # ham
fp = 0 # classed as spam, but is ham
fn = 0 # classed as ham, but is spam

# for each testing example

for i in range(0, len(testing_attributes[:,0])) :

    # perform neural network prediction (i.e. classification)

    # (to get around some kind of OpenCV python interface bug, vertically stack the
    #  example with a second row of zeros of the same size and type which is ignored).

    sample = np.vstack((testing_attributes[i,:],
                        np.zeros(len(testing_attributes[i,:])).astype(np.float32)));

    retrval,output_layer_responses = nnetwork.predict(sample);

    # the class label c (result) is the index of the most
    # +ve of the output layer responses (from the first of the two examples in the stack)

    result = np.argmax(output_layer_responses[0]);

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
