import cv2
import face_recognition
from flask import Flask, render_template, Response

app = Flask(__name__)

# Known face encodings and names
known_face_encodings = []
known_face_names = []

# Load images of known faces and encode them
def load_faces():
    known_person1_image = face_recognition.load_image_file("faces_dataset/person1.jpg")
    known_person2_image = face_recognition.load_image_file("faces_dataset/person2.jpg")
    known_person3_image = face_recognition.load_image_file("faces_dataset/person3.jpg")
    known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
    known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
    known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]

    known_face_encodings.append(known_person1_encoding)
    known_face_encodings.append(known_person2_encoding)
    known_face_encodings.append(known_person3_encoding)

    known_face_names.append("Cains")
    known_face_names.append("Reji")
    known_face_names.append("Murali")

# Capture video and process frames
def gen_frames():
    video_capture = cv2.VideoCapture(0)
    previously_detected_faces = set()  # To store names of previously detected faces
    all_detected_faces = set()  # To track faces during the session
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        current_detected_faces = set()  # Temporary set to track faces detected in the current frame

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            # If the face is not already printed in this frame, print it
            if name not in all_detected_faces:
                print(f"Face detected: {name}")
                all_detected_faces.add(name)
            
            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            current_detected_faces.add(name)
        
        # Remove faces from all_detected_faces that are no longer in the current frame
        all_detected_faces.difference_update(current_detected_faces)

        # Encode frame to send to browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask route to show video stream
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    load_faces()  # Load the faces when the app starts
    app.run(debug=True)
