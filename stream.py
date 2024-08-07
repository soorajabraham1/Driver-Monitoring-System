import cv2 as cv
import numpy as np
import mediapipe as mp
from code.volume import volume_control

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Load YOLO
net = cv.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")

# Get layer names
layer_names = net.getLayerNames()

# Get the output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load the COCO class labels
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def open_len(arr):
    y_arr = [y for _, y in arr]
    return max(y_arr) - min(y_arr)

def get_eye_landmarks(all_landmarks, eye_indices):
    return all_landmarks[eye_indices]

def draw_eye_landmarks(frame, left_eye, right_eye):
    cv.polylines(frame, [left_eye], True, (0, 255, 0), 1, cv.LINE_AA)
    cv.polylines(frame, [right_eye], True, (0, 255, 0), 1, cv.LINE_AA)

def display_eye_heights(frame, max_left, len_left, max_right, len_right):
    cv.putText(img=frame, text=f'Max: {max_left} Left Eye: {len_left}', fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
    cv.putText(img=frame, text=f'Max: {max_right} Right Eye: {len_right}', fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))

def check_drowsiness(len_left, max_left, len_right, max_right, drowsy_frames):
    if len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1:
        drowsy_frames += 1
    else:
        drowsy_frames = 0
    return drowsy_frames

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

def process_frame(frame, face_mesh, img_w, img_h, max_left, max_right, drowsy_frames):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        right_eye, left_eye = all_landmarks[RIGHT_EYE], all_landmarks[LEFT_EYE]
        draw_eye_landmarks(frame, left_eye, right_eye)
        len_left, len_right = open_len(left_eye), open_len(right_eye)
        max_left, max_right = max(max_left, len_left), max(max_right, len_right)
        display_eye_heights(frame, max_left, len_left, max_right, len_right)
        drowsy_frames = check_drowsiness(len_left, max_left, len_right, max_right, drowsy_frames)
        if drowsy_frames > 20:
            cv.putText(frame, 'ALERT', (200, 300), 0, 3, (0, 255, 0), 3)
    return max_left, max_right, drowsy_frames

def main():
    mp_face_mesh = mp.solutions.face_mesh
    
    cap = cv.VideoCapture(0)
    drowsy_frames, max_left, max_right = 0, 0, 0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            max_left, max_right, drowsy_frames = process_frame(frame, face_mesh, img_w, img_h, max_left, max_right, drowsy_frames)
            frame = detect_objects(frame, net, output_layers, classes, "cell phone")
            frame = volume_control(frame)
            cv.imshow('img', frame)
            if cv.waitKey(1) == ord('q'): break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
