#!/usr/bin/env python3.7

import numpy
from scipy.spatial import distance

#POINTS: top -> 1, 2, bottom -> 5, 4
def getEAR(eye):
    vert1 = distance.euclidean(eye[1], eye[5])
    vert2 = distance.euclidean(eye[2], eye[4])
    horiz = distance.euclidean(eye[0], eye[3])
    return (vert1 + vert2) / (2.0 * horiz)