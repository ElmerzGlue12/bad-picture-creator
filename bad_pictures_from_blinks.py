

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import queue
import numpy as np
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
    help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.2,
    help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
    help="the number of consecutive frames the eye must be below the threshold")
ap.add_argument("-d", "--pictureDelay", type = float, default=9,
	help="delay between blink detected to picture taken")
ap.add_argument("-e", "--lowerEAR", type = float, default=0.18,
	help="lower ear vetting range")
ap.add_argument("-g", "--upperEAR", type = float, default=0.22,
	help="upper ear vetting range")
ap.add_argument("-z", "--numCapturedPhotos", type = float, default=9,
	help="number of photos to be catured before next layer of analysis")
ap.add_argument("-i", "--displayInfo", type = bool, default=False,
	help="Option to display EAR, eye trace and photo count on video")

# finds the greatest rate of change in the EAR
def earDerivative(beforeBlink, afterBlink):
    # holds a combination of the before and after blink frames
    frames = []

    # extracts the frames

    for i in range(beforeBlink.qsize()):
        frames.append(beforeBlink.get())

    for i in range(afterBlink.qsize()):
        frames.append(afterBlink.get())

    # holds the change in the ear between the n and n+1 frame
    derivative = []

    for i in range(frames.__len__() - 1):
        derivative.append(abs(frames[i][0] - frames[i+1][0]))

    returnFrames = [], []

    for i in range(10):
        returnFrames[0].append(frames[derivative.index(max(derivative))][1])
        returnFrames[1].append(frames[derivative.index(max(derivative))][0])
        frames.pop(derivative.index(max(derivative)))
        derivative.pop(derivative.index(max(derivative)))

    return returnFrames

def runFrames(vs, detector, predictor, TOTAL, ear):
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame = vs.read()

    frame = imutils.resize(frame, width=1000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # draw the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
    # if args["displayInfo"]:
    #     cv2.putText(frame, "Photos: {}".format(TOTAL), (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #     cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    return (ear, frame)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def main() :
    # FINAL list of images to send to next step
    worstPhotos = []

    args = vars(ap.parse_args())
    EYE_AR_THRESH = args['threshold']
    EYE_AR_CONSEC_FRAMES = args['frames']

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    #initialize queue that holds the frames and ear before and after the blink
    beforeBlink = queue.Queue()
    afterBlink = queue.Queue()


    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    print("[INFO] print q to quit...")
    if args['video'] == "camera":
        vs = VideoStream(src=0).start()
        fileStream = False
    else:
        vs = FileVideoStream(args["video"]).start()
        fileStream = True
   
    time.sleep(1.0)
    
    # loop over frames from the video stream
    while len(worstPhotos) < args["numCapturedPhotos"]:
        try:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            key = 0

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                if args["displayInfo"]:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    # adds a delay after detecting the blink before taking the photo
                    # for i in range(args["pictureDelay"]):
                    # frame = vs.read()
                    # frame = imutils.resize(frame, width=450)
                    # cv2.imwrite("bad_photo.jpg", frame)


                    # empties the queue
                    afterBlink.empty()

                    # saves the blink frame
                    afterBlink.put((ear, frame))

                    # saves the next frames
                    for i in range(10):
                        if len(detector(gray, 0)) > 0:
                            try:
                                afterBlink.put(runFrames(vs, detector, predictor, TOTAL, ear))
                            except:
                                pass

                    # frames from the derivative method
                    derFrames = (earDerivative(beforeBlink, afterBlink))

                    # fig = plt.figure(figsize=(4, 8))
                    # columns = 1
                    # rows = 5
                    # for i in range(1, columns * rows + 1):
                    #     img = derFrames[0][i-1]
                    #     fig.add_subplot(rows, columns, i)
                    #     plt.imshow(img)
                    # plt.show()

                    # worstPhotos.append(derFrames[0][2])
                    # worstPhotos.append(derFrames[0][3])
                    # worstPhotos.append(derFrames[0][4])
                    #
                    # derFrames[0].pop(2)
                    # derFrames[0].pop(2)
                    # derFrames[0].pop(2)

                    if derFrames[1][args["pictureDelay"]] < args["upperEAR"]+0.01 and derFrames[1][args["pictureDelay"]] > args["lowerEAR"]-0.01:
                        worstPhotos.append(derFrames[0][args["pictureDelay"]])
                        derFrames[0].pop(args["pictureDelay"])
                        derFrames[1].pop(args["pictureDelay"])
                        TOTAL += 1
                        print(TOTAL)

                    # vets bad bad images
                    i = 0
                    while(i < len(derFrames[1])):
                        if derFrames[1][i] > args["upperEAR"] or derFrames[1][i] < args["lowerEAR"]:
                            derFrames[0].pop(i)
                            derFrames[1].pop(i)
                        else:
                            i += 1

                    for photo in derFrames[0]:
                        worstPhotos.append(photo)
                        TOTAL += 1
                        print(TOTAL)

                    # fig = plt.figure(figsize=(4, 8))
                    # columns = 1
                    # rows = len(derFrames[0])
                    # for i in range(1, columns * rows + 1):
                    #     img = derFrames[0][i - 1]
                    #     fig.add_subplot(rows, columns, i)
                    #     plt.imshow(img)
                    # plt.show()



                # elif ear > 0.45:
                #     worstPhotos.append(frame)

                # otherwise, the eye aspect ratio is not below the blink 69
                # threshold
                else:
                    # removes the oldest queue item if 10 frames have already been saved
                    if beforeBlink.qsize() >= 20:
                        beforeBlink.get()

                    # adds to the queue of frames before the blink
                    beforeBlink.put((ear,frame))

                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1

                    # reset the eye frame counter
                    COUNTER = 0

                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                    if args["displayInfo"]:
                        cv2.putText(frame, "Photos: {}".format(TOTAL), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # show the frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF
        except:
            pass

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # fig = plt.figure(figsize=(4, 8))
    # columns = 3
    # rows = int(len(worstPhotos)/3)
    # for i in range(1, columns * rows + 1):
    #     img = worstPhotos[i - 1]
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
if __name__ == '__main__' :
    main()
