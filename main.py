from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)

#start recording the webcam (0 is default)
camera = cv2.VideoCapture(0)

#import sample picture and learn how to recognize
christopher_image = face_recognition.load_image_file("photo_examples/Christopher P./csp.jpeg")
#train sample picture
christopher_face_encoding = face_recognition.face_encodings(christopher_image)[0]
#create array of known face encodings and their names
known_face_encodings = [
    christopher_face_encoding
]
known_face_names = [
    "Christopher"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
def gen_frames():

    while True:
        ret0, frame = camera.read()

        if not ret0:
            break
        else:

                #make it smaller 
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                #convert to rgb color
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                #get the face locations from the frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                # get the face encodings from the frame
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names= []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
        
        process_this_frame = not process_this_frame

     # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)