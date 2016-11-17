#####################################################################

# Example : k-Means clustering
# basic illustrative python script

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgement - entirely based on the example at:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv
# as of 2/12/14

#####################################################################

# N.B. we are also relyng on matplotlib being installed

import numpy as np
import cv2
from matplotlib import pyplot as plt

#####################################################################

# generate from random (X,Y) points around with a Guassian distribution
# around two cluster centres with variance as specified

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))

# combine them and convert to np.float32

XY = np.vstack((X,Y))
XY = np.float32(XY)

# define termination criteria and apply kmeans() with k = 2
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags to initialize with random centres

flags = cv2.KMEANS_RANDOM_CENTERS;

# perform k-means clustering

compactness,label,center = cv2.kmeans(XY,2,None,criteria,10, flags);

# Now separate the data by cluster label, Note the flatten()

C1 = XY[label.ravel()==0]
C2 = XY[label.ravel()==1]

# Plot the data
plt.scatter(C1[:,0],C1[:,1])
plt.scatter(C2[:,0],C2[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Attribute X'),plt.ylabel('Atribute Y')
plt.show()
