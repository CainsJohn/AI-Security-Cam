import dlib
import cv2

# Load the detector and predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Cains\Desktop\vscode\main_proj\shape_predictor_68_face_landmarks.dat")
# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect faces in the image
    faces = detector(frame, 1)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(frame, face)

        # Draw a rectangle around the face
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks (optional)
        for p in landmarks.parts():
            cv2.circle(frame, (p.x, p.y), 2, (0, 0, 255), -1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()