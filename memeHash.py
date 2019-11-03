from random import randrange
import cv2
import numpy as np


def openFile(file):
    lineList = []
    for line in open(file):
        items = line.split(',')
        while len(items) < 2:
            items.append('')
        lineList.append((items[0], items[1]))
    return lineList

def getRandomMeme(lineList):
    return (lineList[randrange(len(lineList))])


angerMemes = openFile('angry.txt')
joyMemes = openFile('joy.txt')
sorrowMemes = openFile('sorrow.txt')
surpriseMemes = openFile('surprise.txt')

def getEmotionListName(dictionary):
   l = [dictionary['joy:'], dictionary['sorrow:'], dictionary['anger:'], dictionary['surprise:']]
   temp = l.index(max(l))
   if temp == 0:
       return joyMemes
   elif temp == 1:
       return sorrowMemes
   elif temp == 2:
       return angerMemes
   else:
       return surpriseMemes

def memeCreator(dictionary):
    emotionListName = getEmotionListName(dictionary)
    return getRandomMeme(emotionListName)
    
