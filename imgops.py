#!/usr/bin/env python3.7

# external library imports
import numpy
import cv2
import imutils
from imutils import face_utils
import dlib
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import memeHash as meme

# internal imports
import faceops

# get detector objects from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

font = ImageFont.truetype('impact.ttf', size=38)

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
        EARs = (faceops.getEAR(left), faceops.getEAR(right))
        EAR_list.append((EARs[0] + EARs[1]) / 2.0)

    if len(EAR_list) == 0:
        EAR_list.append(float('nan'))
    
    return EAR_list

def getMARs(face_list):
    MAR_list = []
    for face in face_list:
        mouth = face[mStart:mEnd]
        MAR_list.append(faceops.getMAR(mouth))
    if len(MAR_list) == 0:
        MAR_list.append(0)
    return MAR_list

def writeMeme(image, top, bottom):
    top = top.upper()
    bottom = bottom.upper()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pilImage)
    w, h = draw.textsize(top, font=font)
    w1, h1 = draw.textsize(bottom, font=font)
    W, H = image.shape[1], image.shape[0]
    textX = (W-w)/2
    textY = 55
    textX1 = (W-w1)/2
    textY1 = H-100
    draw.text((textX+2, textY+2), top, fill='black', font=font)
    draw.text((textX-2, textY+2), top, fill='black', font=font)
    draw.text((textX+2, textY-2), top, fill='black', font=font)
    draw.text((textX-2, textY-2), top, fill='black', font=font)
    draw.text((textX, textY), top, fill='white', font=font)

    draw.text((textX1+2, textY1+2), bottom, fill='black', font=font)
    draw.text((textX1-2, textY1+2), bottom, fill='black', font=font)
    draw.text((textX1+2, textY1-2), bottom, fill='black', font=font)
    draw.text((textX1-2, textY1-2), bottom, fill='black', font=font)
    draw.text((textX1, textY1), bottom, fill='white', font=font)
    cvImageRGB = np.array(pilImage)
    cvImage = cv2.cvtColor(cvImageRGB, cv2.COLOR_RGB2BGR)

    return cvImage


def getMemeBuffer(dictList, buffer):
    memes = []
    for d in dictList:
        img = buffer[d["id"]]
        (top, bottom) = meme.memeCreator(d)
        img = writeMeme(img, top, bottom)
        memes.append(img)
    return memes
