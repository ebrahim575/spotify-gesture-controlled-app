import cv2
import mediapipe as mp
import time
import numpy as np
import apple_script as spotify_app


# ===== Configuration =====
def init_config():
    config = {
        # Window parameters
        'WINDOW_WIDTH': 900,
        'WINDOW_HEIGHT': 450,
        'WINDOW_X': 500,
        'WINDOW_Y': 0,

        # Gesture detection parameters
        'Y_SIMILARITY_THRESHOLD': 0.05,
        'X_SIMILARITY_THRESHOLD': 0.05,
        'PAUSE_THRESHOLD_TIME': 0.5,

        # UI state
        'show_data': True,

        # Button parameters
        'BUTTON_X': 10,
        'BUTTON_Y': 10,
        'BUTTON_WIDTH': 100,
        'BUTTON_HEIGHT': 30,

        # MediaPipe setup
        'mp_hands': mp.solutions.hands,
    }

    config['hands'] = config['mp_hands'].Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    return config


def init_landmark_names():
    return [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]


# ===== Mouse Callback =====
def mouse_callback(event, x, y, flags, param):
    config = param['config']
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within button bounds
        if (config['BUTTON_X'] <= x <= config['BUTTON_X'] + config['BUTTON_WIDTH'] and
                config['BUTTON_Y'] <= y <= config['BUTTON_Y'] + config['BUTTON_HEIGHT']):
            config['show_data'] = not config['show_data']
            print(f"Data display: {'On' if config['show_data'] else 'Off'}")


# ===== Hand Analysis Functions =====
def calculate_scale_factor(landmarks):
    """Calculate scale factor based on hand size in frame"""
    wrist = landmarks[0]  # WRIST
    middle_tip = landmarks[12]  # MIDDLE_FINGER_TIP
    hand_height = abs(middle_tip[1] - wrist[1])
    return 0.2 / max(hand_height, 0.1)


def check_fingertips_y_similarity(landmarks):
    """Check if index, middle, and ring fingertips have similar Y values"""
    fingertip_indices = [8, 12, 16]  # Index, Middle, Ring tips (excluding pinky)
    y_values = [landmarks[i][1] for i in fingertip_indices]
    max_y_diff = max(y_values) - min(y_values)

    scale_factor = calculate_scale_factor(landmarks)
    threshold = 0.05 * scale_factor

    is_similar = max_y_diff <= threshold
    print(f"Fingertips Y-similarity: {'PASS' if is_similar else 'FAIL'} "
          f"(diff: {max_y_diff:.3f}, threshold: {threshold:.3f})")
    return is_similar


def check_finger_x_alignment(landmarks, finger_indices, finger_name):
    """Check if points along a finger have similar X values"""
    x_values = [landmarks[i][0] for i in finger_indices]
    max_x_diff = max(x_values) - min(x_values)

    scale_factor = calculate_scale_factor(landmarks)
    threshold = 0.05 * scale_factor

    is_aligned = max_x_diff <= threshold
    print(f"{finger_name} finger X-alignment: {'PASS' if is_aligned else 'FAIL'} "
          f"(diff: {max_x_diff:.3f}, threshold: {threshold:.3f})")
    return is_aligned


def check_pause_gesture(landmarks):
    """Check if hand is in pause gesture position"""
    # Check all fingertips Y similarity (excluding pinky)
    y_similar = check_fingertips_y_similarity(landmarks)

    # Check each finger's X alignment
    fingers = [
        ([5, 6, 7, 8], "Index"),
        ([9, 10, 11, 12], "Middle"),
        ([13, 14, 15, 16], "Ring"),
        ([17, 18, 19, 20], "Pinky")
    ]

    finger_alignments = [
        check_finger_x_alignment(landmarks, indices, name)
        for indices, name in fingers
    ]

    all_conditions_met = y_similar and all(finger_alignments)
    print(f"Overall pause gesture detection: {'PASS' if all_conditions_met else 'FAIL'}")

    return all_conditions_met


# ===== Processing Functions =====
def process_frame(frame, config):
    """Process a single frame and return the resized and flipped version"""
    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (config['WINDOW_WIDTH'] - 400, config['WINDOW_HEIGHT']))  # Changed from 300 to 400
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return frame_resized, frame_rgb


def calculate_velocities(current_landmarks, previous_landmarks, time_diff):
    """Calculate velocities for all landmarks"""
    if previous_landmarks is None or time_diff <= 0:
        return np.zeros_like(current_landmarks)
    return (current_landmarks - previous_landmarks) / time_diff


# ===== Visualization Functions =====
def draw_toggle_button(frame, config):
    """Draw the toggle button for data display"""
    cv2.rectangle(frame,
                  (config['BUTTON_X'], config['BUTTON_Y']),
                  (config['BUTTON_X'] + config['BUTTON_WIDTH'],
                   config['BUTTON_Y'] + config['BUTTON_HEIGHT']),
                  (0, 255, 0),
                  2)
    cv2.putText(frame,
                'Toggle Data',
                (config['BUTTON_X'] + 5, config['BUTTON_Y'] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1)
    return frame


def draw_landmark_numbers(frame, landmarks):
    """Draw landmark numbers on the hand"""
    h, w = frame.shape[:2]
    for id, lm in enumerate(landmarks):
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.putText(frame, str(id), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame


def format_landmark_text(id, landmark_name, x, y, velocity):
    """Format landmark text with consistent spacing"""
    # Format each component with fixed width
    id_text = f"{id:2d}"
    name_text = f"{landmark_name[:10]:<10}"
    x_text = f"X: {x:6.3f}"
    y_text = f"Y: {y:6.3f}"
    v_text = f"V: {velocity:6.3f}"

    return f"{id_text}: {name_text} | {x_text} | {y_text} | {v_text}"


def draw_data_panel(frame, landmarks, velocities, show_data, landmark_names):
    """Draw the data panel showing landmark information"""
    if not show_data:
        return frame

    panel_width = 300
    data_background = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)

    for id, (lm, velocity) in enumerate(zip(landmarks, velocities)):
        text = format_landmark_text(
            id,
            landmark_names[id],
            lm[0],
            lm[1],
            np.linalg.norm(velocity)
        )

        cv2.putText(data_background, text,
                    (10, 20 + id * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1)

    return np.hstack((frame, data_background))


# ===== Main Function =====
def main():
    # Initialize configuration and variables
    config = init_config()
    landmark_names = init_landmark_names()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize window and mouse callback
    cv2.namedWindow("Gesture Control")
    cv2.moveWindow("Gesture Control", config['WINDOW_X'], config['WINDOW_Y'])
    cv2.setMouseCallback("Gesture Control", mouse_callback, {'config': config})

    # Initialize tracking variables
    previous_landmarks = None
    previous_time = None
    pause_gesture_start_time = None
    pause_triggered = False

    print("Starting gesture detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        frame_resized, frame_rgb = process_frame(frame, config)
        current_time = time.time()

        # Draw toggle button
        frame_resized = draw_toggle_button(frame_resized, config)

        results = config['hands'].process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

                time_diff = current_time - previous_time if previous_time else 0
                velocities = calculate_velocities(current_landmarks, previous_landmarks, time_diff)

                frame_resized = draw_landmark_numbers(frame_resized, current_landmarks)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_resized, hand_landmarks, config['mp_hands'].HAND_CONNECTIONS)

                if check_pause_gesture(current_landmarks):
                    if pause_gesture_start_time is None:
                        pause_gesture_start_time = current_time
                    elif (current_time - pause_gesture_start_time >= config['PAUSE_THRESHOLD_TIME']
                          and not pause_triggered):
                        spotify_app.toggle_playback()
                        pause_triggered = True
                else:
                    pause_gesture_start_time = None
                    pause_triggered = False

                previous_landmarks = current_landmarks
                previous_time = current_time

                # Add hand landmarks to frame
                frame_resized = draw_landmark_numbers(frame_resized, current_landmarks)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_resized, hand_landmarks, config['mp_hands'].HAND_CONNECTIONS)

        # Handle data panel visibility regardless of hand presence
        if config['show_data']:
            # Create empty data panel if no hand is detected
            if not results.multi_hand_landmarks:
                data_background = np.zeros((frame_resized.shape[0], 300, 3), dtype=np.uint8)
                frame_resized = np.hstack((frame_resized, data_background))
            else:
                frame_resized = draw_data_panel(
                    frame_resized, current_landmarks, velocities, config['show_data'], landmark_names)
        else:
            # If data display is toggled off, don't add the panel
            frame_resized = frame_resized

        cv2.imshow("Gesture Control", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()