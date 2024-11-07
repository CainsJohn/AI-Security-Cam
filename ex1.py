import face_recognition

# Load an image file
image = face_recognition.load_image_file(r"C:\Users\Cains\Pictures\Camera Roll\person2.jpg")


# Find all face locations
face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s) in this image.")
