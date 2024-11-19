import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a named window and set its properties
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)  # Allow the window to be resizable
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_TOPMOST, 1)  # Keep the window always on top

# Define desired window size and screen dimensions
window_width, window_height = 200, 120  # Maintain 5:3 ratio for the window size
screen_width, screen_height = 3024, 1964  # Mac M1 Pro 14-inch screen resolution

# Adjust for window decoration offsets
# On macOS, window borders/title bars need to be considered.
border_offset = 50  # Adjust this as needed based on actual offset

# Calculate position for top-right corner
x_position = screen_width - window_width - border_offset  # Ensure margin from right edge
y_position = 30  # Account for macOS title bar height

# Resize and move the window
cv2.resizeWindow("Camera Feed", window_width, window_height)
cv2.moveWindow("Camera Feed", x_position, y_position)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the small window size
    frame = cv2.resize(frame, (window_width, window_height))

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()