#!/usr/bin/env python3.7

import cv2
import time
import imutils
from imutils import face_utils
import imgops
import numpy as np

# Get video from Webcam
cap = cv2.VideoCapture(0)

print("Hello! We are about to start recording! Please press ESC to stop recording...")

time.sleep(2.0)

buffer = []

# 'record' data - load into buffer list
while len(buffer) < 100:  # only record 100 frames
    _, img = cap.read()
    buffer.append(img)
    cv2.imshow('img', img)
    if cv2.waitKey(66) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

# get EARs for each frame
EARs = []
for img in buffer:
    entry = imgops.getEARs(imgops.getFaceFeatures(img))
    EARs.append(entry)

arr = np.array(EARs)
#arr[np.isnan(arr)] = np.inf
indexOrder = np.argsort(arr, axis=0)
img = buffer[indexOrder[0][0]]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
