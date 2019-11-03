#!/usr/bin/env python3.7

import numpy
from scipy.spatial import distance

#POINTS: top -> 1, 2, bottom -> 5, 4
def getEAR(eye):
    vert1 = distance.euclidean(eye[1], eye[5])
    vert2 = distance.euclidean(eye[2], eye[4])
    horiz = distance.euclidean(eye[0], eye[3])
    return (vert1 + vert2) / (2.0 * horiz)

# POINTS: horiz -> 12-16, vert -> 13-19, 14-18, 15-17
def getMAR(mouth):
    vert1 = distance.euclidean(mouth[13], mouth[19])
    vert2 = distance.euclidean(mouth[14], mouth[18])
    vert3 = distance.euclidean(mouth[17], mouth[17])
    horiz = distance.euclidean(mouth[12], mouth[16])
    return (vert1 + vert2) / (3.0 * horiz)