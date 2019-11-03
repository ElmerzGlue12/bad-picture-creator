#!/usr/bin/env python3.7
import cv2
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
import dlib
from scipy.spatial import distance

import faceops
import imgops

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# To capture video from webcam. 
cap = WebcamVideoStream(src=0).start()
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while True:
    # Read the frame
    img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    face_list = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        face_list.append(shape)

    if len(face_list) > 0:
        left_eye = face_list[0][lStart:lEnd]
        right_eye = face_list[0][rStart:rEnd]
        left_ear = faceops.getEAR(left_eye)
        right_ear = faceops.getEAR(right_eye)
        MAR = imgops.getMARs(face_list)[0]
        for (x, y) in face_list[0][mStart:mEnd]:
            img[y, x] = [255,255,255]
    else:
        left_eye = []
        right_eye = []
        left_ear = 0
        right_ear = 0

    cv2.putText(img, "EAR: " + str((left_ear+right_ear)/2), (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.putText(img, "EAM: " + str(MAR), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xff == 27:
        break
# Release the VideoCapture object
cap.stop()
cv2.destroyAllWindows()