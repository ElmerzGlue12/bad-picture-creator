import io
import os
import json
from pprint import pprint

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

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

joy_scale = 1
sorrow_scale = 3
anger_scale = 3
surprise_scale= 10

for img in os.listdir('google_images'):
    # The name of the image file to annotate

    # Loads the image into memory
    with io.open('google_images\\' + img, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content = content)

    # Performs label detection on the image file
    response = client.face_detection(image=image)
    faces = response.face_annotations

    bad_factor = 0

    print('Face Annotate:')
    for label in faces:
        print(label)
        print(possibility[label.joy_likelihood][0])

    bad_factor = get_bad_factor(faces)

    print(bad_factor)
