import cv2
import face_recognition

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    face_list = face_recognition.face_landmarks(img)
    if len(face_list) > 0:
        left_eye = face_list[0]['left_eye']
        right_eye = face_list[0]['right_eye']
    else:
        left_eye = []
        right_eye = []

    for (x, y) in left_eye:
        img[y,x] = [255,255,255]

    for (x, y) in right_eye:
        img[y,x] = [255,255,255]

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()