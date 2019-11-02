#!/usr/bin/env python3.7
import cv2
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
import dlib
from scipy.spatial import distance


import eyeutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# To capture video from webcam. 
cap = WebcamVideoStream(src=0).start()
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
        left_ear = eyeutils.getEAR(left_eye)
        right_ear = eyeutils.getEAR(right_eye)
    else:
        left_eye = []
        right_eye = []
        left_ear = 0
        right_ear = 0

    for (x, y) in left_eye:
        img[y,x] = [0,255,0]

    cv2.putText(img, "EAR: " + str((left_ear+right_ear)/2), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    for (x, y) in right_eye:
        img[y,x] = [0,255,0]

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
# Release the VideoCapture object
cap.stop()
cv2.destroyAllWindows()