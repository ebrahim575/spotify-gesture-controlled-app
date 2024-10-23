# Spotify Gesture Control

This project uses computer vision to control Spotify playback through hand gestures. It utilizes OpenCV and MediaPipe for hand tracking and gesture recognition, and the Spotify API for playback control.

## Demo

Check out this demo of the Spotify gesture control app in action:

[Spotify Gesture Control Demo](spotify_gesture_demo.mp4)

## Features

- Next track: Swipe hand from left to right
- Previous track: Swipe hand from right to left
- Play/Pause: Hold palm facing the camera for 0.5 seconds

## Prerequisites

- Python 3.7+
- Spotify Premium account
- Spotify Developer account 

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/spotify-gesture-control.git
   cd spotify-gesture-control
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Spotify API credentials:
   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
   - Create a new app
   - Set the redirect URI to `http://localhost:8888/callback`
   - Note your Client ID and Client Secret

5. Create a `creds.py` file in the project root with your Spotify API credentials:
   ```python
   CLIENT_ID = 'your_client_id_here'
   CLIENT_SECRET = 'your_client_secret_here'
   ```

## Usage

1. Activate the virtual environment (if not already activated):
   ```
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. Use hand gestures in front of your camera to control Spotify playback:
   - Swipe left to right for next track
   - Swipe right to left for previous track
   - Hold palm facing camera for play/pause

4. Press 'q' to quit the application

## For Developers

### Pulling from origin/main

To update your local repository with the latest changes from the main branch:

1. Ensure you're on your local main branch:
   ```
   git checkout main
   ```

2. Pull the latest changes:
   ```
   git pull origin main
   ```

### Adding new packages

If you add new Python packages to the project:

1. Install the package:
   ```
   pip install package_name
   ```

2. Update the requirements.txt file:
   ```
   pip freeze > requirements.txt
   ```

3. Commit the updated requirements.txt file

### Virtual Environment

Always work within the virtual environment to keep dependencies isolated. To activate it:

```
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```