import spotipy
from spotipy.oauth2 import SpotifyOAuth
from creds import CLIENT_ID, CLIENT_SECRET
import json

REDIRECT_URI = 'http://localhost:8888/callback'  # This should match your app settings
SCOPE = 'user-read-playback-state user-modify-playback-state'

# Set up SpotifyOAuth
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
)

# Create Spotify client
sp = spotipy.Spotify(auth_manager=sp_oauth)

def next_track():
    try:
        sp.next_track()
        print("Successfully skipped to next track")
    except Exception as e:
        print(f"Error skipping to next track: {e}")

def previous_track():
    try:
        sp.previous_track()
        print("Returned to previous track")
    except Exception as e:
        print(f"Error returning to previous track: {e}")

def toggle_playback():
    try:
        current_playback = sp.current_playback()
        if current_playback is not None and current_playback['is_playing']:
            sp.pause_playback()
            print("Playback paused")
        else:
            sp.start_playback()
            print("Playback started")
    except Exception as e:
        print(f"Error toggling playback: {str(e)}")

def start_playback():
    try:
        sp.start_playback()
        print("Starting playback successfully.")
    except Exception as e:
        print(f"Error starting playback: {str(e)}")

def restart_playback():
    try:
        sp.seek_track(0)
        print("Restarted current track")
    except Exception as e:
        print(f"Error restarting playback: {str(e)}")
