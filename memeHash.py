from random import randrange
import cv2
import numpy as np


def openFile(file):
    lineList = [line.rstrip('\n') for line in open(file)]
    return lineList

def getRandomMeme(lineList):
    return (lineList[randrange(len(lineList))])


angerMemes = openFile('angry.txt')
joyMemes = openFile('joy.txt')
sorrowMemes = openFile('sorrow.txt')
surpriseMemes = openFile('surprise.txt')

def getEmotionListName(dictionary):
   list = [dictionary['joy:'], dictionary['sorrow:'], dictionary['anger:'], dictionary['surprise:']]
   temp = list.index(max(list))
   if temp == 0:
       return 'joyMemes'
   elif temp == 1:
       return 'sorrowMemes'
   elif temp == 2:
       return 'angerMemes'
   else:
       return 'surpriseMemes'

def memeCreator(dictionary):
    emotionListName = getEmotionListName(dictionary)
    return getRandomMeme(emotionListName)
    
