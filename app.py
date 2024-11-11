import cv2
import dlib
import numpy as np
import os


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/haarcascades/shape_predictor_68_face_landmarks.dat")
face_rec_model_path = "assets/haarcascades/dlib_face_recognition_resnet_model_v1.dat"
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


known_face_encodings = []
known_face_names = []

def load_known_faces():
    known_faces_dir = "known_faces"
    
    print("loading known faces...")
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"cant load face: {filename}")
                continue
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_image)
            
            if len(faces) == 0:
                print(f"no face in {filename} ")
                continue
            
            shape = predictor(rgb_image, faces[0])
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
            known_face_encodings.append(np.array(face_descriptor))
            known_face_names.append(os.path.splitext(filename)[0])
    
    print(f"had load {len(known_face_encodings)} faces")

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
            
            if len(faces) > 0:
                shape = predictor(rgb_image, faces[0])
                face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
                known_face_encodings.append(np.array(face_descriptor))
                known_face_names.append(os.path.splitext(filename)[0])
    

def save_unknown_face(frame, face_location, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    cv2.imwrite(f"unknown_faces/{name}.jpg", face_image)

def recognize_face_from_camera():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("cant open camera")
        return

    unknown_counter = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("cant read pictures")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = detector(rgb_small_frame)

        for face_location in face_locations:
            top = face_location.top() * 4
            right = face_location.right() * 4
            bottom = face_location.bottom() * 4
            left = face_location.left() * 4

            shape = predictor(rgb_small_frame, face_location)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)

            matches = []
            for known_face_encoding in known_face_encodings:
                match = np.linalg.norm(known_face_encoding - np.array(face_descriptor))
                matches.append(match)
            
            name = "Unknown"
            if len(matches) > 0 and min(matches) < 0.6:
                name = known_face_names[matches.index(min(matches))]

            if name == "Unknown":
                unknown_name = f"Unknown_{unknown_counter}"
                save_unknown_face(frame, (top, right, bottom, left), unknown_name)
                unknown_counter += 1
                
                unknown_image = cv2.imread(f"unknown_faces/{unknown_name}.jpg")
                unknown_rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
                unknown_faces = detector(unknown_rgb)
                if len(unknown_faces) > 0:
                    unknown_shape = predictor(unknown_rgb, unknown_faces[0])
                    unknown_encoding = face_rec_model.compute_face_descriptor(unknown_rgb, unknown_shape)
                    known_face_encodings.append(np.array(unknown_encoding))
                    known_face_names.append(unknown_name)
                    name = unknown_name

            color = (0, 0, 255) if name.startswith("Unknown") else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    load_unknown_faces()
    recognize_face_from_camera()