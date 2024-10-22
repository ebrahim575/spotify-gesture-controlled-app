import cv2
import mediapipe as mp
import time
import spotify_app
import numpy as np

# Window parameters (adjust as needed)
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
WINDOW_X = 100  # X position of the window
WINDOW_Y = 100  # Y position of the window

# Gesture detection parameters
FINGER_DISTANCE_THRESHOLD_X = 0.08  # Increased X threshold for index and middle finger
FINGER_DISTANCE_THRESHOLD_Y = 0.08  # Increased Y threshold for index and middle finger
FINGER_DISTANCE_THRESHOLD = 0.05  # Keep the original threshold for other checks
VELOCITY_THRESHOLD = 0.5  # Adjust this value to change velocity sensitivity
USE_VELOCITY = True  # Flag to enable/disable velocity check
MOVEMENT_DISTANCE_THRESHOLD = 0.1  # Minimum distance to move from left to right

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to send the next song command
def next_song():
    print('Executing the next song gesture...')
    spotify_app.skip_to_next_track()

# Function to check for the specific hand pose
def check_hand_pose(landmarks, velocities):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP.value]

    # Check if index and middle finger tips are close, using the new thresholds
    index_middle_close = (abs(index_tip[0] - middle_tip[0]) < FINGER_DISTANCE_THRESHOLD_X and
                          abs(index_tip[1] - middle_tip[1]) < FINGER_DISTANCE_THRESHOLD_Y)

    # Check if middle and ring finger tips are far apart in Y
    middle_ring_apart = abs(middle_tip[1] - ring_tip[1]) > FINGER_DISTANCE_THRESHOLD

    # Check velocity of index and middle fingers
    index_velocity = np.linalg.norm(velocities[mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
    middle_velocity = np.linalg.norm(velocities[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value])
    sufficient_velocity = index_velocity > VELOCITY_THRESHOLD and middle_velocity > VELOCITY_THRESHOLD

    return index_middle_close and middle_ring_apart and sufficient_velocity

# Set up the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera initialized successfully.")

# Landmark names
landmark_names = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Initialize previous landmarks and time
previous_landmarks = None
previous_time = None
last_gesture_time = 0
debounce_time = 1.0  # 1 second debounce
gesture_start_x = None

print("Starting main loop...")

# Create named window and move it to desired position
cv2.namedWindow("Gesture Control")
cv2.moveWindow("Gesture Control", WINDOW_X, WINDOW_Y)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (WINDOW_WIDTH - 300, WINDOW_HEIGHT))  # Leave space for data on the right
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Create a black background for the data
    data_background = np.zeros((WINDOW_HEIGHT, 300, 3), dtype=np.uint8)
    
    current_time = time.time()

    # Default threshold state
    threshold_state = "Threshold Not Met"
    threshold_color = (0, 255, 0)  # Green

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            
            if previous_landmarks is not None and previous_time is not None:
                time_diff = current_time - previous_time
                if time_diff > 0:
                    velocities = (current_landmarks - previous_landmarks) / time_diff
                else:
                    velocities = np.zeros_like(current_landmarks)
                
                for id, (lm, velocity) in enumerate(zip(current_landmarks, velocities)):
                    # Display landmark data with green, bold, thick text
                    cv2.putText(data_background, f"{id}: {landmark_names[id][:10]}", (10, 20 + id * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(data_background, f"X:{lm[0]:.2f} Y:{lm[1]:.2f} V:{np.linalg.norm(velocity):.2f}", (10, 40 + id * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw landmark number on the hand
                    h, w, _ = frame_resized.shape
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.putText(frame_resized, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if check_hand_pose(current_landmarks, velocities):
                    index_tip_x = current_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value][0]
                    
                    if gesture_start_x is None:
                        gesture_start_x = index_tip_x
                    else:
                        movement_distance = abs(index_tip_x - gesture_start_x)
                        if movement_distance > MOVEMENT_DISTANCE_THRESHOLD:
                            if (current_time - last_gesture_time) > debounce_time:
                                if index_tip_x < gesture_start_x:  # Right to left movement
                                    spotify_app.skip_to_previous_track()
                                    print("Previous track gesture detected")
                                else:  # Left to right movement
                                    spotify_app.skip_to_next_track()
                                    print("Next track gesture detected")
                                last_gesture_time = current_time
                            gesture_start_x = None  # Reset the start position after gesture is detected
                else:
                    gesture_start_x = None  # Reset the start position if hand pose is not maintained

            mp.solutions.drawing_utils.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            previous_landmarks = current_landmarks
            previous_time = current_time

    # Display threshold state
    cv2.putText(frame_resized, threshold_state, (10, WINDOW_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, threshold_color, 2)

    # Combine the frame and data background
    combined_frame = np.hstack((frame_resized, data_background))

    cv2.imshow("Gesture Control", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
