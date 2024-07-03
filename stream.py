import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define a function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for drowsiness detection
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 48

# Initialize frame counters
COUNTER = 0
ALARM_ON = False

def main():
    st.title("Drowsiness Detection System")
    
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()
    
    global COUNTER, ALARM_ON

    stop_button = st.button('Stop', key='stop_button')  # Unique key for the button

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Ignoring empty camera frame.")
            continue

        # Convert the frame color from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect face and landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Extract the eye landmarks
                left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

                # Convert normalized coordinates to pixel values
                ih, iw, _ = frame.shape
                left_eye = [(int(lm.x * iw), int(lm.y * ih)) for lm in left_eye]
                right_eye = [(int(lm.x * iw), int(lm.y * ih)) for lm in right_eye]

                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Visualize the landmarks
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

                # Check if EAR is below the threshold
                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            st.warning("Drowsiness Alert!")
                else:
                    COUNTER = 0
                    ALARM_ON = False

        # Display the resulting frame
        stframe.image(frame, channels='BGR')

        if stop_button:
            break

    cap.release()

if __name__ == '__main__':
    main()
