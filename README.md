## Purpose
This project was created for VandyHacks VI in November 2019 in 36 hours. Our original idea revolved around taking in a video stream and flagging all "bad" photos, then having either a funny collection of bad photos of someone or a useful collection of "good" photos. We elected to go for the funny route. We had those features completed after only about 20 hours, so we decided it would be fun to go further along the comedy route and try to make memes out of the bad photos, which resulted in us using Google's Vision API to detect emotion in those images, then producing text based on those emotions.

## Dependencies
- OpenCV (Python bindings) >= 3.4
- dlib >= 19.0
- imutils
- scipy
- Access to Google Vision API (including credential file) with Python libraries

## Usage
### Realtime Processing
`py bad_pictures_from_blinks.py`
-This executable will grab frames from the webcam and detect blinking frames in real time. It then uses the rate of change of eye aspect ratio to find mid-blink photos and save them, then quit and add text based on Google Vision emotion guesses after 10-12 photos depending on the time taken. Not recommended for computers with low- to mid-range processing power, as face detection and blink detection in real time requires fast image processing. 

### Post-processing
`py find_bad_photos_post.py`
-This executable will grab frames from the webcam for about 5 seconds continuously and finds the photos with the smallest eye aspect ratio and the largest mouth aspect ratio. It will then get emotion data for each of these photos and assign text to each depending on the emotion. 

## Debugging and Proof-of-Concept
These files were developed during our research for proof of concept and testing. We considered removing them, but elected to leave them here in case they can help anyone understand how this application's image processing works. 

- detect_eyes_realtime.py
  - This file is a simple python executable that grabs frames from the webcam and detects faces and eyes. Writes average EAR of the two eyes to the frame.
  - Usage: `py detect_eyes_realtime.py`
  
- recognize_faces.py
  - This file is a simple python executable that detects multiple faces from an image and boxes them.
  - Formerly face_detection.py, renamed to avoid module naming conflicts
  - Usage: `py recognize_faces.py`

- eye_detect.py
  - This file detects multiple faces and eyes from a webcam stream and boxes them. 
  - Usage: `py eye_detect.py`
