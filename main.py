import os
import cv2
import face_recognition

def recoglef():
    known_face_encodings = []
    known_face_names = []

    # Check if image files exist before loading them
    if not os.path.exists("faces_dataset/person1.jpg"):
        print("Error: 'faces_dataset/person1.jpg' not found.")
        return
    if not os.path.exists("faces_dataset/person2.jpg"):
        print("Error: 'faces_dataset/person2.jpg' not found.")
        return
    if not os.path.exists("faces_dataset/person3.jpg"):
        print("Error: 'faces_dataset/person3.jpg' not found.")
        return

    # Load images using the correct module name
    known_person1_image = face_recognition.load_image_file("faces_dataset/person1.jpg")
    known_person2_image = face_recognition.load_image_file("faces_dataset/person2.jpg")
    known_person3_image = face_recognition.load_image_file("faces_dataset/person3.jpg")

    # Get face encodings for each person, if faces are detected
    known_person1_encoding = face_recognition.face_encodings(known_person1_image)
    if not known_person1_encoding:
        print("No face detected in person1.jpg")
        return
    known_person1_encoding = known_person1_encoding[0]

    known_person2_encoding = face_recognition.face_encodings(known_person2_image)
    if not known_person2_encoding:
        print("No face detected in person2.jpg")
        return
    known_person2_encoding = known_person2_encoding[0]

    known_person3_encoding = face_recognition.face_encodings(known_person3_image)
    if not known_person3_encoding:
        print("No face detected in person3.jpg")
        return
    known_person3_encoding = known_person3_encoding[0]

    # Add the encodings and names to the lists
    known_face_encodings.append(known_person1_encoding)
    known_face_encodings.append(known_person2_encoding)
    known_face_encodings.append(known_person3_encoding)

    known_face_names.append("Cains")
    known_face_names.append("Arjun")
    known_face_names.append("Reji")
    
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
    
        # Detect faces and encode them in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"
        
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                name = "unknown"
            
            # Draw a rectangle around the face and put the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show the frame with the rectangles and names
        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture the frame when 'c' is pressed
            try:
                image_name = f"frameleft.jpg"
                cv2.imwrite(image_name, frame)
                print(f"Image captured and saved as: {image_name}")
            except Exception as e:
                print(f"Error capturing image: {e}")
    
        elif key == ord('q'):  # 'q' key pressed to quit
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recoglef()
