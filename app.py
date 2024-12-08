import cv2
import dlib
import numpy as np
import os

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/haarcascades/shape_predictor_68_face_landmarks.dat")

# Initialize face recognition model
face_rec_model_path = "assets/haarcascades/dlib_face_recognition_resnet_model_v1.dat"
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Lists to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces from the "known_faces" directory
def load_known_faces():
    known_faces_dir = "known_faces"
    print("Loading known faces...")
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot load face: {filename}")
                continue
            
            # Convert the image to RGB and detect faces
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_image)
            
            # Ensure at least one face is detected
            if len(faces) == 0:
                print(f"No face found in {filename}")
                continue
            
            # Extract facial landmarks and compute face descriptor
            shape = predictor(rgb_image, faces[0])
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
            known_face_encodings.append(np.array(face_descriptor))
            known_face_names.append(os.path.splitext(filename)[0])
    
    print(f"Loaded {len(known_face_encodings)} faces.")

# Load unknown faces from the "unknown_faces" directory
def load_unknown_faces():
    unknown_faces_dir = "unknown_faces"
    if not os.path.exists(unknown_faces_dir):
        os.makedirs(unknown_faces_dir)
    
    for filename in os.listdir(unknown_faces_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(unknown_faces_dir, filename)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_image)
            
            # Detect faces and extract descriptors if any faces are found
            if len(faces) > 0:
                shape = predictor(rgb_image, faces[0])
                face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
                known_face_encodings.append(np.array(face_descriptor))
                known_face_names.append(os.path.splitext(filename)[0])

# Save unknown faces to the "unknown_faces" directory
def save_unknown_face(frame, face_location, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    cv2.imwrite(f"unknown_faces/{name}.jpg", face_image)

# Perform real-time face recognition using the camera
def recognize_face_from_camera():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Cannot open camera.")
        return

    unknown_counter = 0

    while True:
        # Read a frame from the camera
        ret, frame = video_capture.read()
        if not ret:
            print("Cannot read frames.")
            break

        # Resize the frame for faster processing and convert it to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = detector(rgb_small_frame)

        for face_location in face_locations:
            # Scale the face location back to the original frame size
            top = face_location.top() * 4
            right = face_location.right() * 4
            bottom = face_location.bottom() * 4
            left = face_location.left() * 4

            # Extract face descriptor
            shape = predictor(rgb_small_frame, face_location)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)

            # Compare the descriptor with known faces
            matches = []
            for known_face_encoding in known_face_encodings:
                match = np.linalg.norm(known_face_encoding - np.array(face_descriptor))
                matches.append(match)
            
            name = "Unknown"
            # Determine if there is a match
            if len(matches) > 0 and min(matches) < 0.6:
                name = known_face_names[matches.index(min(matches))]

            # Handle unknown faces
            if name == "Unknown":
                unknown_name = f"Unknown_{unknown_counter}"
                save_unknown_face(frame, (top, right, bottom, left), unknown_name)
                unknown_counter += 1
                
                # Process and store the unknown face
                unknown_image = cv2.imread(f"unknown_faces/{unknown_name}.jpg")
                unknown_rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
                unknown_faces = detector(unknown_rgb)
                if len(unknown_faces) > 0:
                    unknown_shape = predictor(unknown_rgb, unknown_faces[0])
                    unknown_encoding = face_rec_model.compute_face_descriptor(unknown_rgb, unknown_shape)
                    known_face_encodings.append(np.array(unknown_encoding))
                    known_face_names.append(unknown_name)
                    name = unknown_name

            # Draw bounding box and label around the face
            color = (0, 0, 255) if name.startswith("Unknown") else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # press 'q' to exit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    load_unknown_faces()
    recognize_face_from_camera()
