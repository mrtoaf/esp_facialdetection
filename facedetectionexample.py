# program uses Haar Cascade filters in order to detect faces in an image
# thanks to NeuralNine for the bulk of the code!
# a Haar Cascade is "an Object Detection Algorithm used to identify faces in an image or a real time video"
# read some more: https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08

import cv2
import pathlib

# get the xml datafile for a Haar Cascade
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# set the cascade classifier as the Haar cascade file we just got 
clf = cv2.CascadeClassifier(str(cascade_path))

# set up the default camera (0) i hope this is the webacam lmaoo
camera = cv2.VideoCapture(0)

#this is a test

while True:
    # screencap!
    ret, frame = camera.read()
    # so BGR is reverse RGB xD, make it gray bc we dont need color for face recognition, much simpler
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        # higher number is stricter detection
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        # use openCV to plot a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(50) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()




