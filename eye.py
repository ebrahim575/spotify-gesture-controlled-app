import cv2
import mediapipe as mp
import time
import spotify_app

# Global variables and initializations
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

window_width = 432
window_height = 324
window_x = 1079
window_y = -80

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
    spotify_app.next_track()

def previous_song():
    print('Executing the previous song gesture...')
    spotify_app.skip_to_previous_track()

def draw_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_face_landmarks(face_landmarks, frame_resized):
    # Get eye landmarks
    left_eye = [face_landmarks.landmark[145], face_landmarks.landmark[159]]  # Upper and lower landmarks for left eye
    right_eye = [face_landmarks.landmark[374], face_landmarks.landmark[386]]  # Upper and lower landmarks for right eye

    # Calculate eye aspect ratio (EAR)
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    draw_text(frame_resized, "Face detected", (20, 60), color=(0, 255, 0))
    draw_text(frame_resized, f"Left EAR: {left_ear:.3f}", (20, 90))
    draw_text(frame_resized, f"Right EAR: {right_ear:.3f}", (20, 120))

    return left_ear, right_ear, time.time()

def calculate_ear(eye):
    # Calculate the vertical distance between the eye landmarks
    return abs(eye[0].y - eye[1].y)

def detect_eye_closure(left_ear, right_ear, current_time, last_action_time, frame_resized):
    blink_threshold = 0.025  # Adjust this value based on your observations
    blink_duration_threshold = 0.5  # Minimum duration for a blink to trigger an action

    if current_time - last_action_time > debounce_time:
        if left_ear < blink_threshold and right_ear >= blink_threshold:
            previous_song()
            draw_text(frame_resized, "Previous Song - Left Eye Closed!", (20, 480), color=(255, 0, 0))
            print("Previous Song - Left Eye Closed!")
            return True, current_time
        elif right_ear < blink_threshold and left_ear >= blink_threshold:
            next_song()
            draw_text(frame_resized, "Next Song - Right Eye Closed!", (20, 480), color=(0, 255, 255))
            print("Next Song - Right Eye Closed!")
            return True, current_time

    return False, last_action_time

def main_loop():
    cap = initialize_camera()
    last_action_time = 0

    print("Starting main loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (window_width, window_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear, right_ear, current_time = process_face_landmarks(face_landmarks, frame_resized)
                
                action_triggered, last_action_time = detect_eye_closure(left_ear, right_ear, current_time, last_action_time, frame_resized)

                mp_drawing.draw_landmarks(
                    image=frame_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        cv2.imshow("Eye Closure Detection", frame_resized)
        cv2.moveWindow("Eye Closure Detection", window_x, window_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main_loop()
