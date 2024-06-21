import cv2
import dlib
import numpy as np
from incabin_utils import eye_aspect_ratio, head_pose_estimation, findEncodings
from volume import volume_control
from imutils import face_utils
import face_recognition
import os
from ui import myUi
from textbox import draw_filled_rounded_rectangle





overlay_image = cv2.imread('icons/profile.jpg', cv2.IMREAD_UNCHANGED)  # Load with alpha channel
# Resize the overlay image if needed
overlay_image = cv2.resize(overlay_image, (50, 50))  # Resize to desired size

# Load the pre-trained shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Constants for eye aspect ratio (EAR) and consecutive frames for drowsiness detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5
DISTRACTION_COUNTER = 0
# Initialize the frame counter
COUNTER = 0
DISP_COUNTER=0

# Extract the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Load YOLO
net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")

# Get layer names
layer_names = net.getLayerNames()

# Get the output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load the COCO class labels
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the pre-trained facial expression recognition model
expression_net = cv2.dnn.readNetFromONNX("models/emotion-ferplus-8.onnx")
expression_list = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# Load user images and encode faces
path = 'users'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])



# Encode the known faces
encoded_face_train = findEncodings(images)

# Variables for tracking recognized faces
authorized_user = None
face_not_found_counter = 0
face_not_found_threshold = 30  # Number of frames to wait before resetting authorized user

# Start the video stream
cap = cv2.VideoCapture(0)



while True:
    success, frame = cap.read()
    if not success:
        break
    
    height, width, channels = frame.shape
    # Convert frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    if authorized_user is None:
        # Detect faces and compute encodings for face recognition
        
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        
        if faces_in_frame:
            face_not_found_counter = 0  # Reset the counter if faces are found
            cv2.rectangle(frame, (20, 30), (300, 75), (0, 0, 255),  cv2.FILLED)
            cv2.putText(frame, "Unauthorized Person", (25,  60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)

                if matches[matchIndex]:
                    name = classNames[matchIndex].lower()
                    authorized_user = name

                    # Draw rectangle and name on the recognized face
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    break
    else:
        # Check if the authorized user is still in the frame
        faces_in_frame = face_recognition.face_locations(imgS)
        if faces_in_frame:
            face_not_found_counter = 0  # Reset the counter if faces are found
            if DISP_COUNTER<15:
                DISP_COUNTER+=1
            for faceloc in faces_in_frame:
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                if DISP_COUNTER<15:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, authorized_user, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                if DISP_COUNTER==15:
                    frame=myUi(frame,overlay_image)
                    #cv2.rectangle(frame, (65, 30), (200, 75), (0, 255, 0),  cv2.FILLED)
                    frame = draw_filled_rounded_rectangle(frame, (65, 30), (200, 75),  (0, 255, 0), 10, 1)
                    cv2.putText(frame, authorized_user, (75,  60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
        else:
            #DISP_COUNTER=0
            # Increment the counter if no faces are found
            face_not_found_counter += 1
            if face_not_found_counter > face_not_found_threshold:
                authorized_user = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                #cv2.putText(frame, "DROWSINESS ALERT!", (400, 30),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                frame = draw_filled_rounded_rectangle(frame, (400, 80), (640, 115), (0, 0, 255), 10, 1)           
                #cv2.rectangle(frame, (400, 80), (640, 115), (0, 0, 255),  cv2.FILLED)
                cv2.putText(frame, "DROWSINESS ALERT!", (440,  100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            COUNTER = 0

        # Head pose estimation
        (nose_end_point2D, pitch, yaw, roll) = head_pose_estimation(shape, frame)
        p1 = (int(shape[30][0]), int(shape[30][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        #cv2.line(frame, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
         
        # Check for distraction
        if abs(yaw) > 700 and abs(pitch) > 140:#30,20, or
            DISTRACTION_COUNTER+=1
            if DISTRACTION_COUNTER > 11:
                #cv2.putText(frame, "DISTRACTION ALERT!", (10, 60),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                frame = draw_filled_rounded_rectangle(frame, (400, 40), (640, 75), (0, 0, 255), 10, 1)
                #cv2.rectangle(frame, (400, 40), (640, 75), (0, 0, 255),  cv2.FILLED)
                cv2.putText(frame, "DISTRACTION ALERT!", (440,  60),cv2.LINE_AA, 0.5, (255, 255, 255), 2)
        else:
            DISTRACTION_COUNTER = 0
        
        # Detect facial expression
        face_image = gray[rect.top():rect.bottom(), rect.left():rect.right()]
        face_image = cv2.resize(face_image, (64, 64))  # Resize to 64x64 pixels
        face_blob = cv2.dnn.blobFromImage(face_image, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)  # Ensure the image is resized properly
        expression_net.setInput(face_blob)
        expression_preds = expression_net.forward()
        expression = expression_list[np.argmax(expression_preds[0])]

        #cv2.putText(frame, f"Expression: {expression}", (rect.left(), rect.top() - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.rectangle(frame, (400, 80), (640, 115), (0, 255, 0),  cv2.FILLED)
        frame = draw_filled_rounded_rectangle(frame, (400, 0), (640, 35), (0, 255, 0), 10, 1)
        cv2.putText(frame, f"Emotion: {expression}", (440,  20),cv2.LINE_AA, 0.5, (255, 255, 255), 2)
        # Draw facial landmarks on the frame
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
    
    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen
    class_ids = []
    confidences = []
    boxes = []
    target_class = "cell phone" 
    for out in outs:
        for detection in out: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == target_class:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                frame = draw_filled_rounded_rectangle(frame, (400, 120), (640, 150), (0, 0, 255), 10, 1)
                cv2.putText(frame, "Keep Phone Away", (440,  140),cv2.LINE_AA, 0.5, (255, 255, 255), 2)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green for detected objects
            #if label == "cell phone":
                #color = (0, 0, 255)  # Red for cell phone

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    frame = volume_control(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
