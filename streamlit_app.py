import streamlit as st
import cv2
import dlib
import numpy as np
from code.incabin_utils import head_pose_estimation, findEncodings, process_faces, process_eyes_and_head, detect_expression, detect_objects
from code.volume import volume_control
from imutils import face_utils
import os
import uuid

st.title("Real-Time Video Processing")

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5

# Initialize counters
COUNTER = 0
DISP_COUNTER = 0
DISTRACTION_COUNTER = 0
face_not_found_counter = 0
face_not_found_threshold = 30

# Load models and data
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

expression_net = cv2.dnn.readNetFromONNX("models/emotion-ferplus-8.onnx")
expression_list = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

path = 'users'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encoded_face_train = findEncodings(images)

# Variables for tracking recognized faces
authorized_user = None

# Start the video stream
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()

# Create buttons
stop_button_key = str(uuid.uuid4())
exit_button_key = str(uuid.uuid4())

stop_button = st.button("Stop", key=stop_button_key)
exit_button = st.button("Exit", key=exit_button_key)

# Main loop
while cap.isOpened():
    if stop_button or exit_button:
        break

    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    frame, authorized_user, face_not_found_counter, DISP_COUNTER = process_faces(
        frame, imgS, authorized_user, face_not_found_counter, DISP_COUNTER, face_not_found_threshold, encoded_face_train, classNames
    )

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        frame, COUNTER, DISTRACTION_COUNTER = process_eyes_and_head(shape, frame, COUNTER, DISTRACTION_COUNTER)

        # Add a check to ensure face_image is not empty
        try:
            frame = detect_expression(gray, rect, expression_net, expression_list, frame)
        except cv2.error as e:
            st.warning("No face detected. Continuing to next frame.")
            continue

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    frame = detect_objects(frame, net, output_layers, classes, "cell phone")
    frame = volume_control(frame)

    # Display the frame using Streamlit
    frame_placeholder.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
