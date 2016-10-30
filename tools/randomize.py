#####################################################################

# Example : randomize loaded CVS file and write out to file
# basic illustrative python script

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import numpy as np

from random import shuffle

########### Load Data Set - Example

# load full data set (unsplit)

reader=csv.reader(open("input.data","rt", encoding='ascii'),delimiter=',')


entry_list = []

for row in reader:
        entry_list.append(row)

########### randomize (different order for every file loaded)
# N.B. to randomize attributes / labels together - append into single np array
# with one {attribute/label} pair together, then shuffle

shuffle(entry_list)

########### Write Data Set - Example

writer = csv.writer(open("output.data", "wt", encoding='ascii'), delimiter=',')
writer.writerows(entry_list)

#####################################################################
