#!/usr/bin/env python3.7

import cv2
import time
import imutils
from imutils import face_utils
import imgops
import numpy as np
from google_vision import get_results_for_image

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
MARs = []
for img in buffer:
    features = imgops.getFaceFeatures(img)
    EAR = imgops.getEARs(features)
    MAR = imgops.getMARs(features)
    EARs.append(EAR)
    MARs.append(MAR)

# convert to numpy array for quicker sorting
EARray = np.array(EARs)
MARray = np.array(MARs)
indexOrderEyes = np.argsort(EARray, axis=0)
indexOrderMouth = np.argsort(MARray, axis=0)
eyeBuf = indexOrderEyes[0:5][0:5]
mouthBuf = indexOrderMouth[-5:][-5:]

for i in eyeBuf:
    cv2.imshow('img', buffer[i[0]])
    cv2.waitKey(0)
for i in mouthBuf:
    cv2.imshow('img', buffer[i[0]])
    cv2.waitKey(0)
cv2.destroyAllWindows()

eyeBufList = []
mouthBufList = []

for i in eyeBuf:
    results = get_results_for_image(buffer[i[0]])
    results["id"] = i[0]
    eyeBufList.append(results)
for i in mouthBuf:
    results = get_results_for_image(buffer[i[0]])
    results["id"] = i[0]
    mouthBufList.append(results)

print(eyeBufList)
print(mouthBufList)
