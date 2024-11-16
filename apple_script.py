import os
import sys

def next_track():
    try:
        print("Attempting next track via AppleScript...")
        os.system("""osascript -e 'tell application "Spotify" to next track'""")
        print("Successfully triggered next track")
    except Exception as e:
        print(f"Error: {e}")

def previous_track():
    try:
        print("Attempting previous track via AppleScript...")
        os.system("""osascript -e 'tell application "Spotify" to previous track'""")
        print("Successfully triggered previous track")
    except Exception as e:
        print(f"Error: {e}")

def toggle_playback():
    try:
        print("Attempting to toggle playback via AppleScript...")
        os.system("""osascript -e 'tell application "Spotify" to playpause'""")
        print("Successfully toggled playback")
    except Exception as e:
        print(f"Error: {e}")
