import cv2
import mediapipe as mp
import time
import spotify_app

# Global variables and initializations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
debug_mode = 0

window_width = 432
window_height = 324
window_x = 1079
window_y = -80

gesture_threshold = 0.009
gesture_speed_threshold = 0.009
gesture_duration_threshold = 0.15
debounce_time = 1

def initialize_camera():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    print("Camera initialized successfully.")
    return cap

def next_song():
    print('Executing the next song gesture...')
    spotify_app.skip_to_next_track()

def draw_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_hand_landmarks(hand_landmarks, frame_resized, previous_data):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    current_index_x = index_finger_tip.x
    current_middle_x = middle_finger_tip.x
    current_time = time.time()

    draw_text(frame_resized, "Hand detected", (20, 60), color=(0, 255, 0))
    draw_text(frame_resized, f"Index X: {current_index_x:.3f}", (20, 120))
    draw_text(frame_resized, f"Middle X: {current_middle_x:.3f}", (20, 180))

    return current_index_x, current_middle_x, current_time

def calculate_movement(current_data, previous_data, frame_resized):
    current_index_x, current_middle_x, current_time = current_data
    previous_index_x, previous_middle_x, previous_time = previous_data

    time_diff = current_time - previous_time
    delta_index_x = current_index_x - previous_index_x
    delta_middle_x = current_middle_x - previous_middle_x

    if time_diff > 0:
        index_speed = delta_index_x / time_diff
        middle_speed = delta_middle_x / time_diff
        avg_speed = (abs(index_speed) + abs(middle_speed)) / 2
    else:
        index_speed = middle_speed = avg_speed = 0

    draw_text(frame_resized, f"Movement: {'Right' if delta_index_x > 0 else 'Left'}", (20, 240))
    draw_text(frame_resized, f"Index Speed: {index_speed:.3f}", (20, 300))
    draw_text(frame_resized, f"Middle Speed: {middle_speed:.3f}", (20, 360))
    draw_text(frame_resized, f"Avg Speed: {avg_speed:.3f}", (20, 420))

    return delta_index_x, delta_middle_x, avg_speed

def detect_gesture(delta_index_x, delta_middle_x, avg_speed, gesture_start_time, current_time, frame_resized):
    if delta_index_x > gesture_threshold and delta_middle_x > gesture_threshold and avg_speed > gesture_speed_threshold:
        if gesture_start_time is None:
            gesture_start_time = current_time
        elif current_time - gesture_start_time > gesture_duration_threshold:
            next_song()
            draw_text(frame_resized, "Next Song Gesture Detected!", (20, 480), color=(0, 255, 255))
            print("Next Song Gesture Detected and Executed!")
            return True, current_time
    else:
        gesture_start_time = None
    return False, gesture_start_time

def main_loop():
    cap = initialize_camera()
    previous_index_x = previous_middle_x = previous_time = None
    gesture_start_time = None
    gesture_detected = False
    last_gesture_time = 0

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
                current_data = process_hand_landmarks(hand_landmarks, frame_resized, (previous_index_x, previous_middle_x, previous_time))
                
                if all(previous_data is not None for previous_data in (previous_index_x, previous_middle_x, previous_time)):
                    delta_index_x, delta_middle_x, avg_speed = calculate_movement(current_data, (previous_index_x, previous_middle_x, previous_time), frame_resized)
                    
                    current_time = current_data[2]
                    if not gesture_detected and (current_time - last_gesture_time) > debounce_time:
                        gesture_detected, gesture_start_time = detect_gesture(delta_index_x, delta_middle_x, avg_speed, gesture_start_time, current_time, frame_resized)
                        if gesture_detected:
                            last_gesture_time = current_time
                    elif gesture_detected and (current_time - last_gesture_time) > debounce_time:
                        gesture_detected = False

                previous_index_x, previous_middle_x, previous_time = current_data

                mp.solutions.drawing_utils.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Control", frame_resized)
        cv2.moveWindow("Gesture Control", window_x, window_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main_loop()
