import io
import os
import json
import cv2
from pprint import pprint

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson


# Define scales for each emotion
joy_scale = 1
sorrow_scale = 3
anger_scale = 3
surprise_scale= 10


# Returns bad factors for each image
def get_bad_factor(faces):
    bad_factor = 0
    for label in faces:
        bad_factor += possibility[label.joy_likelihood][1] * joy_scale
        bad_factor += possibility[label.sorrow_likelihood][1] * sorrow_scale
        bad_factor += possibility[label.anger_likelihood][1] * anger_scale
        bad_factor += possibility[label.surprise_likelihood][1] * surprise_scale
    return bad_factor


credentials = service_account.Credentials.from_service_account_file('vandyhacks_googlecred_2019.json')
# Instantiates a client
client = vision.ImageAnnotatorClient(credentials=credentials)

possibility = (('UNKNOWN', 0), ('VERY_UNLIKELY', 1), ('UNLIKELY',2),
               ('POSSIBLE',3), ('LIKELY',4), ('VERY_LIKELY',5))

# gets the emotion scores for each image in folder
def get_results_for_image(imageInput):
	content = cv2.imencode('.jpg', imageInput)[1].tostring()
	
	image = vision.types.Image(content = content)

    # Performs label detection on the image file
	response = client.face_detection(image=image)
	faces = response.face_annotations

	bad_factor = 0
	imageScores = {}

	for label in faces:
		# add emotion values to dict
		imageScores["joy:"] = possibility[label.joy_likelihood][1]
		imageScores["anger:"] = possibility[label.anger_likelihood][1]
		imageScores["sorrow:"] = possibility[label.sorrow_likelihood][1]
		imageScores["surprise:"] = possibility[label.surprise_likelihood][1]
		imageScores["bad factor:"] = get_bad_factor(faces)

	return imageScores



if __name__ == '__main__':
	for img in os.listdir('google_images'):

		# 
		cont = cv2.imread('google_images/' + img)
		faceDict = get_results_for_image(cont)
		print(faceDict)
