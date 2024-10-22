import cv2
import mediapipe as mp
import time
import spotify_app

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
debug_mode = 0

# Toggles for gesture behavior
use_restart_playback = 0  # Toggle to control restart playback feature
use_next_track = 1  # Toggle to control next track feature

# Window size and position parameters
window_width = 216
window_height = 162
window_x = 1295  # X position for top-right corner (adjust as needed)
window_y = 0    # Y position for top-right corner (adjust as needed)

# Function to send the next song command
def next_song():
    print('Executing the next song gesture...')
    spotify_app.skip_to_next_track()

# Function to send the previous song command
def previous_song():
    print('Executing the previous song gesture...')
    spotify_app.previous_track()

# Function to restart playback
def restart_playback():
    print('Executing the restart playback gesture...')
    spotify_app.restart_playback()

# Set up the webcam
print("Initializing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera initialized successfully.")

# Initialize previous positions and time
previous_index_x = None
previous_middle_x = None
previous_time = None

# Gesture detection parameters
gesture_threshold = 0.009  # Lower value for more sensitivity
gesture_speed_threshold = 0.009  # Lower value for more sensitivity
gesture_duration_threshold = 0.15  # Lower value for quicker detection
gesture_start_time = None
gesture_detected = False
last_gesture_time = 0
debounce_time = 1

# Function to draw text on frame
def draw_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

print("Starting main loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (window_width, window_height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            current_index_x = index_finger_tip.x
            current_middle_x = middle_finger_tip.x
            current_time = time.time()

            draw_text(frame_resized, "Hand detected", (10, 30), color=(0, 255, 0))
            draw_text(frame_resized, f"Index X: {current_index_x:.3f}", (10, 60))
            draw_text(frame_resized, f"Middle X: {current_middle_x:.3f}", (10, 90))

            if previous_index_x is not None and previous_middle_x is not None and previous_time is not None:
                time_diff = current_time - previous_time
                delta_index_x = current_index_x - previous_index_x
                delta_middle_x = current_middle_x - previous_middle_x

                if time_diff > 0:
                    index_speed = delta_index_x / time_diff
                    middle_speed = delta_middle_x / time_diff
                    avg_speed = (abs(index_speed) + abs(middle_speed)) / 2
                else:
                    index_speed = middle_speed = avg_speed = 0

                draw_text(frame_resized, f"Movement: {'Right' if delta_index_x > 0 else 'Left'}", (10, 120))
                draw_text(frame_resized, f"Index Speed: {index_speed:.3f}", (10, 150))
                draw_text(frame_resized, f"Middle Speed: {middle_speed:.3f}", (10, 180))
                draw_text(frame_resized, f"Avg Speed: {avg_speed:.3f}", (10, 210))

                if not gesture_detected and (current_time - last_gesture_time) > debounce_time:
                    if debug_mode:
                        print(f"Debug: delta_index_x: {delta_index_x:.3f}, delta_middle_x: {delta_middle_x:.3f}, avg_speed: {avg_speed:.3f}")
                        print(f"Thresholds: gesture: {gesture_threshold}, speed: {gesture_speed_threshold}")

                    # Detect right-to-left swipe for next song or restart playback
                    if delta_index_x > gesture_threshold and delta_middle_x > gesture_threshold and avg_speed > gesture_speed_threshold:
                        if gesture_start_time is None:
                            gesture_start_time = current_time
                        elif current_time - gesture_start_time > gesture_duration_threshold:
                            # Restart playback if toggle is enabled
                            if use_restart_playback:
                                restart_playback()
                                gesture_detected = True
                                last_gesture_time = current_time
                                draw_text(frame_resized, "Restart Playback Gesture Detected!", (10, 240), color=(255, 255, 0))
                                print("Restart Playback Gesture Detected and Executed!")

                            # Skip to next track if toggle is enabled
                            elif use_next_track:
                                next_song()
                                gesture_detected = True
                                last_gesture_time = current_time
                                draw_text(frame_resized, "Next Song Gesture Detected!", (10, 240), color=(0, 255, 255))
                                print("Next Song Gesture Detected and Executed!")
                    else:
                        gesture_start_time = None
                elif gesture_detected and (current_time - last_gesture_time) > debounce_time:
                    gesture_detected = False

            previous_index_x = current_index_x
            previous_middle_x = current_middle_x
            previous_time = current_time

            mp.solutions.drawing_utils.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame_resized)

    # Move window to the desired location
    cv2.moveWindow("Gesture Control", window_x, window_y)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
