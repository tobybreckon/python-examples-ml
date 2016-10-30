#####################################################################

# Example : select subset of lines from CVS file and write to files
# basic illustrative python script

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np
import math

from random import shuffle

########### Load Data Set - Example

# load full data set (unsplit)

reader=csv.reader(open("input.data","rt", encoding='ascii'),delimiter=',')

entry_list = []

for row in reader:
        entry_list.append(row)

########### Write Data Set - Example

# write first N% of the entries to first file

N = 30.0

writerA = csv.writer(open("outputA.data", "wt", encoding='ascii'), delimiter=',')
writerA.writerows(entry_list[0:int(math.floor(len(entry_list)* (N/100.0)))])

# write the remaining (100-N)% of the entries of the second file

writerB = csv.writer(open("outputB.data", "wt", encoding='ascii'), delimiter=',')
writerB.writerows(entry_list[int(math.floor(len(entry_list)* (N/100.0))):len(entry_list)])

#####################################################################
