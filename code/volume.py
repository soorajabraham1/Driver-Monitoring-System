import cv2
import mediapipe as mp


# Function to detect hand landmarks
def detect_hand_landmarks(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect hand landmarks
    results = hands.process(frame_rgb)
    return results

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start the video stream
#cap = cv2.VideoCapture(0)

# Initialize variables for volume control
prev_y = None
max_y = None
volume = 50  # Starting volume
volume_increment = 30  # Volume change per movement unit
movement_threshold = 30  # Minimum movement threshold for volume change

def volume_control(frame):
    global prev_y, max_y, volume

    # Detect hand landmarks
    results = detect_hand_landmarks(frame)

    # Process the results to extract gesture information
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the hand
            middle_finger_tip = (int(hand_landmarks.landmark[12].x * frame.shape[1]), int(hand_landmarks.landmark[12].y * frame.shape[0]))
            index_finger_tip = (int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))

            # Calculate the vertical movement of the band formed by joining thumb and forefinger
            
            if (abs(middle_finger_tip[1]- index_finger_tip[1]))<12 and 0<middle_finger_tip[0]< 100:
                current_y = (middle_finger_tip[1] + index_finger_tip[1]) // 2
                print(abs(middle_finger_tip[1]- index_finger_tip[1]))

            # Initialize max_y on the first frame
                if max_y is None:
                    max_y = current_y

            # Update max_y if current_y is greater
                if current_y > max_y:
                    max_y = current_y

            # Calculate volume change based on vertical movement
                if prev_y is not None:
                    movement = current_y - prev_y
                    if abs(movement) > movement_threshold:
                        volume += int(movement / abs(movement)) * volume_increment

                # Clip volume within range
                volume = max(0, min(200, volume))
                print(volume)

                # Update previous y coordinate
                prev_y = current_y

            # Draw volume bar on the frame
            
            cv2.rectangle(frame, (50, 200), (70, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (50, 400), (70 , 200+volume), (0, 255, 0), cv2.FILLED)


            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    return frame
        
#     # Display the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
