#!/usr/bin/env python3.7

# external library imports
import numpy
import cv2
import imutils
from imutils import face_utils
import dlib

# internal imports
import eyeutils

# get detector objects from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def getFaceFeatures(img, pyramids=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, pyramids)
    face_list = []

    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        face_list.append(shape)

    return face_list

def getEARs(face_list):
    EAR_list = []
    for face in face_list:
        left = face[lStart:lEnd]
        right = face[rStart:rEnd]
        EARs = (eyeutils.getEAR(left), eyeutils.getEAR(right))
        EAR_list.append((EARs[0] + EARs[1]) / 2.0)

    if len(EAR_list) == 0:
        EAR_list.append(float('nan'))
    
    return EAR_list
