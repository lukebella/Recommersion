# https://www.youtube.com/watch?v=DGeDcxul5Zk

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Scale
from PIL import Image, ImageTk  # Pillow is used for image handling (install with 'pip install pillow') if needed

# Basic setup of main Tkinter window
root = tk.Tk()
root.title("Recommersion - Emotion-Based Music Recommendation")
root.geometry("600x600")
root.resizable(False, False)

# Function placeholders for core functionalities
def start_microphone():
    messagebox.showinfo("Microphone", "Microphone started! Please speak.")
    # Here, add functionality to start speech recognition and detect emotion

def generate_playlist_from_text():
    text_input = text_entry.get()
    if text_input:
        messagebox.showinfo("Generating Playlist", f"Generating playlist based on text input: {text_input}")
        # Add functionality to process text and generate playlist
    else:
        messagebox.showwarning("Warning", "Please enter text to generate playlist.")

def adjust_recommendation():
    # Fetch valence and arousal values from sliders
    valence = valence_slider.get()
    arousal = arousal_slider.get()
    messagebox.showinfo("Adjusting Recommendation", f"Adjusting playlist with valence {valence} and arousal {arousal}")
    # Add functionality to re-compute playlist based on valence/arousal

def play_song():
    # Function to play the song
    messagebox.showinfo("Play", "Playing song")

def pause_song():
    # Function to pause the song
    messagebox.showinfo("Pause", "Pausing song")

def next_song():
    # Function to skip to the next song
    messagebox.showinfo("Next Song", "Skipping to next song")

def previous_song():
    # Function to go to the previous song
    messagebox.showinfo("Previous Song", "Going to previous song")

# === Frame 1: Input Frame ===
input_frame = ttk.LabelFrame(root, text="Input Emotion")
input_frame.pack(pady=10, padx=10, fill="both", expand="yes")

# Microphone button for speech input
mic_button = ttk.Button(input_frame, text="🎤 Speak Emotion", command=start_microphone)
mic_button.grid(row=0, column=0, padx=10, pady=10)

# Text Entry box for text input
text_entry = ttk.Entry(input_frame, width=40)
text_entry.grid(row=0, column=1, padx=10, pady=10)

# Button to submit text input for playlist generation
text_button = ttk.Button(input_frame, text="Generate Playlist from Text", command=generate_playlist_from_text)
text_button.grid(row=0, column=2, padx=10, pady=10)

# === Frame 3: Playlist Frame ===
playlist_frame = ttk.LabelFrame(root, text="Playlist")
playlist_frame.pack(pady=10, padx=10, fill="both", expand="yes")

# Placeholder label for playlist (could be a listbox or Treeview for displaying songs)
playlist_label = ttk.Label(playlist_frame, text="Playlist will be displayed here", padding=10)
playlist_label.pack()

# === Frame 2: Adjustment Frame ===
adjustment_frame = ttk.LabelFrame(root, text="Adjust Emotion - Valence and Arousal")
adjustment_frame.pack(pady=10, padx=10, fill="both", expand="yes")

# Valence slider
valence_label = ttk.Label(adjustment_frame, text="Valence:")
valence_label.grid(row=0, column=0, padx=10, pady=10)
valence_slider = Scale(adjustment_frame, from_=0, to=1000, orient="horizontal", length=200)
valence_slider.grid(row=0, column=1, padx=10, pady=10)

# Arousal slider
arousal_label = ttk.Label(adjustment_frame, text="Arousal:")
arousal_label.grid(row=1, column=0, padx=10, pady=10)
arousal_slider = Scale(adjustment_frame, from_=0, to=1000, orient="horizontal", length=200)
arousal_slider.grid(row=1, column=1, padx=10, pady=10)

# Button to re-generate playlist with adjusted valence/arousal
adjust_button = ttk.Button(adjustment_frame, text="Recompute Playlist", command=adjust_recommendation)
adjust_button.grid(row=2, column=0, columnspan=2, pady=10)

# === Frame 4: Playback Controls Frame ===
controls_frame = ttk.LabelFrame(root, text="Playback Controls")
controls_frame.pack(pady=10, padx=10, fill="both", expand="yes")

# Play, Pause, Previous, Next buttons
play_button = ttk.Button(controls_frame, text="Play", command=play_song)
play_button.grid(row=0, column=0, padx=10, pady=10)

pause_button = ttk.Button(controls_frame, text="Pause", command=pause_song)
pause_button.grid(row=0, column=1, padx=10, pady=10)

previous_button = ttk.Button(controls_frame, text="⏮ Prev", command=previous_song)
previous_button.grid(row=0, column=2, padx=10, pady=10)

next_button = ttk.Button(controls_frame, text="Next ⏭", command=next_song)
next_button.grid(row=0, column=3, padx=10, pady=10)

# Volume slider for volume control
volume_label = ttk.Label(controls_frame, text="Volume:")
volume_label.grid(row=1, column=0, padx=10, pady=10)
volume_slider = Scale(controls_frame, from_=0, to=100, orient="horizontal", length=200)
volume_slider.set(50)  # Setting default volume to 50%
volume_slider.grid(row=1, column=1, columnspan=3, padx=10, pady=10)

# Run the main event loop
root.mainloop()
