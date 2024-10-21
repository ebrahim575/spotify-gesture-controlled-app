# 🎵 Spotify Gesture Control 👋

Control your Spotify playback with simple hand gestures! This application uses computer vision to detect hand movements and translate them into Spotify commands.

## ✨ Features

- 👉 Skip to next track with a right-to-left swipe
- 👈 Restart current track with a left-to-right swipe
- 🖐️ Pause/Play functionality (currently disabled, but code is available)

## 🛠️ Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher installed
- A Spotify Premium account
- A webcam connected to your computer

## 🚀 Getting Started

Follow these steps to get your Spotify Gesture Control up and running:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spotify-gesture-control.git
cd spotify-gesture-control
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Spotify Developer Account

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Create a new application
4. Note down the `Client ID` and `Client Secret`
5. Add `http://localhost:8888/callback` to the Redirect URIs in your app settings

### 4. Configure Credentials

Create a file named `creds.py` in the project directory with the following content:

```python
CLIENT_ID = 'your_client_id_here'
CLIENT_SECRET = 'your_client_secret_here'
```

Replace `'your_client_id_here'` and `'your_client_secret_here'` with the values from your Spotify Developer Dashboard.

### 5. Run the Application

Execute the main script:

```bash
python main.py
```

On first run, you'll be prompted to authorize the application. Follow the instructions in your terminal.

## 🕹️ Usage

Once the application is running:

1. Ensure your webcam has a clear view of your hand
2. To skip to the next track: Swipe your hand from right to left
3. To restart the current track: Swipe your hand from left to right
4. To exit the application: Press 'q' on your keyboard

## 📝 Notes

- Ensure you have an active Spotify session on a device before running the application
- Adjust lighting for better hand detection if necessary
- You can fine-tune gesture sensitivity in the `main.py` file

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) for computer vision capabilities
- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [Spotipy](https://spotipy.readthedocs.io/) for Spotify API integration

---
