import math
from scipy.spatial import distance as dist
import cv2
import numpy as np
import face_recognition
import os
from ui import myUi
from textbox import draw_filled_rounded_rectangle
from imutils import face_utils
# Extract the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

overlay_image = cv2.imread('icons/profile.jpg', cv2.IMREAD_UNCHANGED)  # Load with alpha channel
overlay_image = cv2.resize(overlay_image, (50, 50))  # Resize to desired size

# Constants for eye aspect ratio (EAR) and consecutive frames for drowsiness detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5

# Initialize the frame counter



# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def head_pose_estimation(shape, frame):
    # 3D model points of the face.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    # 2D image points from the detected landmarks.
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")
    # Camera internals
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1)) 
    # SolvePnP to find the 3D pose of the head
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    # Project a 3D point (0, 0, 1000.0) onto the image plane
    (nose_end_point2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    # Calculate the angles
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.degrees(angle[0]) for angle in eulerAngles]  # Extract the single element from the array
    return (nose_end_point2D, pitch, yaw, roll)


def process_faces(frame, imgS, authorized_user, face_not_found_counter, DISP_COUNTER, face_not_found_threshold, encoded_face_train, classNames):
    faces_in_frame = face_recognition.face_locations(imgS)
    if authorized_user is None:
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        if faces_in_frame:
            face_not_found_counter = 0  # Reset the counter if faces are found
            cv2.rectangle(frame, (20, 30), (300, 75), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Unauthorized Person", (25, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)
                if matches[matchIndex]:
                    authorized_user = classNames[matchIndex].lower()
                    break
    else:
        if faces_in_frame:
            face_not_found_counter = 0
            if DISP_COUNTER < 15:
                DISP_COUNTER += 1
            for faceloc in faces_in_frame:
                y1, x2, y2, x1 = [coord * 4 for coord in faceloc]
                if DISP_COUNTER < 15:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, authorized_user, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    frame = myUi(frame, overlay_image)
                    frame = draw_filled_rounded_rectangle(frame, (65, 30), (200, 75), (0, 255, 0), 10, 1)
                    cv2.putText(frame, authorized_user, (75, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            face_not_found_counter += 1
            if face_not_found_counter > face_not_found_threshold:
                authorized_user = None
    return frame, authorized_user, face_not_found_counter, DISP_COUNTER


def process_eyes_and_head(shape, frame, COUNTER, DISTRACTION_COUNTER):
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    if ear < EYE_AR_THRESH:
        COUNTER += 1
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            frame = draw_filled_rounded_rectangle(frame, (400, 80), (640, 115), (0, 0, 255), 10, 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (440, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        COUNTER = 0
    (nose_end_point2D, pitch, yaw, roll) = head_pose_estimation(shape, frame)
    if abs(yaw) > 700 and abs(pitch) > 140:
        DISTRACTION_COUNTER += 1
        if DISTRACTION_COUNTER > 11:
            frame = draw_filled_rounded_rectangle(frame, (400, 40), (640, 75), (0, 0, 255), 10, 1)
            cv2.putText(frame, "DISTRACTION ALERT!", (440, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        DISTRACTION_COUNTER = 0
    return frame, COUNTER, DISTRACTION_COUNTER


def detect_expression(gray, rect, expression_net, expression_list, frame):
    face_image = gray[rect.top():rect.bottom(), rect.left():rect.right()]
    face_image = cv2.resize(face_image, (64, 64))
    face_blob = cv2.dnn.blobFromImage(face_image, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    expression_net.setInput(face_blob)
    expression_preds = expression_net.forward()
    expression = expression_list[np.argmax(expression_preds[0])]
    frame = draw_filled_rounded_rectangle(frame, (400, 0), (640, 35), (0, 255, 0), 10, 1)
    cv2.putText(frame, f"Emotion: {expression}", (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def detect_objects(frame, net, output_layers, classes, target_class):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == target_class:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                frame = draw_filled_rounded_rectangle(frame, (400, 120), (640, 150), (0, 0, 255), 10, 1)
                cv2.putText(frame, "Keep Phone Away", (440, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
