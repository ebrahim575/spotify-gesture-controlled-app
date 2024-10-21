import spotipy
from spotipy.oauth2 import SpotifyOAuth
from creds import CLIENT_ID,CLIENT_SECRET

REDIRECT_URI = 'http://localhost:8888/callback'  # This should match your app settings

SCOPE = 'user-modify-playback-state'

# Set up SpotifyOAuth
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
)

# Create Spotify client
sp = spotipy.Spotify(auth_manager=sp_oauth)

def skip_to_next_track():
    try:
        sp.next_track()
        print("Skipped to next song successfully.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")

def pause_track():
    try:
        sp.pause_playback()
        print("Paused playback successfully.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")

def start_playback():
    try:
        sp.start_playback()
        print("Starting playback successfully.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")

def previous_track():
    try:
        sp.previous_track()
        print("Went back on playback successfully.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")

def restart_playback():
    try:
        sp.seek_track(position_ms=0)
        print("Went back on playback successfully.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")


