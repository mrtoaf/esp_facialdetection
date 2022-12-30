import cv2
import face_recognition
import numpy as np

#start recording the webcam (0 is default)
camera = cv2.VideoCapture(0)

face_locations = face_recognition.load_image_file("edward.png")


cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# set the cascade classifier as the Haar cascade file we just got 
clf = cv2.CascadeClassifier(str(cascade_path))

# set up the default camera (0) i hope this is the webacam lmaoo
camera = cv2.VideoCapture(0)



while True:
    # screencap!
    ret, frame = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = face_recognition.load_image_file(gray)

    face_locations = face_recognition.face_locations(image)

   
    cv2.imshow("Faces", face_locations)

    if cv2.waitKey(50) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



