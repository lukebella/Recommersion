import tkinter as tk
from tkinter import ttk, messagebox, Scale
from vocal_assistant.vocal_assistant import VocalAssistant
from vocal_assistant.emotion.emotion_model import process_func
from vocal_assistant.emotion.predict_emotion import load_trained_model, predict_emotion, return_device
from vocal_assistant.emotion.emotion_model import process_func
import pandas as pd
import numpy as np
import threading
import dask.dataframe as dd
import pickle

class Functions:
    def __init__(self, callback):
        self.callback = callback
        self.vc = VocalAssistant(1)
        threading.Thread(target=self.load_data).start()

    def load_data(self):
        self.device = return_device()
        self.audeering_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.custom_model_name = "model_checkpoint_sampled.pth"
        self.pretrained_model = "facebook/wav2vec2-base"
        self.custom_model, self.custom_processor = load_trained_model(self.device, \
                                                                      self.audeering_model_name, self.pretrained_model)
        self.custom = False
        with open("./data/Songs", "rb") as f:
            self.data = pickle.load(f)
        print("Songs loaded!")
        #self.data = pd.read_pickle("./data/Songs").sample(frac = 0.2, random_state=42).reset_index(drop=True)

        self.callback()

    def start_microphone(self):
        self.vc.talk("What is your mood today?")
        return self.vc.take_vocal_command() 
    
    def process_audio_file(self, file):
        return self.vc.process_audio_file(file)
    
    def predict_valence_arousal(self, file):
        return predict_emotion(self.custom_model, self.device, self.custom_processor, file)[0].tolist() if self.custom else \
               process_func(file, 16000)[0][:2]
    
    def generate_playlist(self, dimensional, cut = 5):
        songs_list = pd.DataFrame({"id": self.data["musicId"], "eucl_dist":self.data[["Valence", "Arousal"]]\
                           .apply(lambda x: np.linalg.norm(x - dimensional), axis=1), "Valence": self.data["Valence"], \
                            "Arousal": self.data["Arousal"],\
                            "title":self.data["title"], "artist": self.data["artist"], "mp3_file":self.data["mp3_file"]})

        return songs_list.sort_values(by="eucl_dist")[:cut]


class Recommersion:
    def __init__(self, root):
        self.root = root
        self.root.title("Recommersion - Emotion-Based Music Recommendation")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        
        self.create_widgets()
        self.loading_label = ttk.Label(self.root, text="Loading dataset, please wait...", font=("Arial", 16))
        self.loading_label.pack(pady=20)
        
        self.functions = None
        self.initialize_functions()

    def initialize_functions(self):
        self.functions = Functions(self.on_data_loaded)

    def on_data_loaded(self):
        # Remove loading message
        self.loading_label.pack_forget()
        self.create_widgets()
        messagebox.showinfo("Data Loaded", "Songs dataset loaded successfully!")

    def create_widgets(self):
        self.create_input_frame()
        self.create_playlist_frame()
        self.create_adjustment_frame()
        self.create_controls_frame()

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Input Emotion")
        input_frame.pack(pady=10, padx=10, fill="both", expand="yes")
        
        ttk.Button(input_frame, text="🎤 Speak Emotion", command=self.start_microphone).grid(row=0, column=0, padx=10, pady=10)
        
        self.text_label= ttk.Label(input_frame, text="Your speech text:", font=("Arial", 16), width=40)
        self.text_label.grid(row=0, column=1, padx=10, pady=10)
        
        #ttk.Button(input_frame, text="Generate Playlist from Text", command=self.generate_playlist_from_text).grid(row=0, column=2, padx=10, pady=10)

    def create_adjustment_frame(self):
        adjustment_frame = ttk.LabelFrame(self.root, text="Adjust Emotion - Valence and Arousal")
        adjustment_frame.pack(pady=10, padx=10, fill="both", expand="yes")

        ttk.Label(adjustment_frame, text="Valence:").grid(row=0, column=0, padx=10, pady=10)
        self.valence_slider = Scale(adjustment_frame, from_=0, to=1000, orient="horizontal", length=200)
        self.valence_slider.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(adjustment_frame, text="Arousal:").grid(row=1, column=0, padx=10, pady=10)
        self.arousal_slider = Scale(adjustment_frame, from_=0, to=1000, orient="horizontal", length=200)
        self.arousal_slider.grid(row=1, column=1, padx=10, pady=10)

        ttk.Button(adjustment_frame, text="Recompute Playlist", command=self.adjust_recommendation).grid(row=2, column=0, columnspan=2, pady=10)

    def create_playlist_frame(self):
        playlist_frame = ttk.LabelFrame(self.root, text="Playlist")
        playlist_frame.pack(pady=10, padx=10, fill="both", expand="yes")
        self.playlist_label = ttk.Label(playlist_frame, text="Playlist will be displayed here", padding=10)
        self.playlist_label.pack()

    def create_controls_frame(self):
        controls_frame = ttk.LabelFrame(self.root, text="Playback Controls")
        controls_frame.pack(pady=10, padx=10, fill="both", expand="yes")

        ttk.Button(controls_frame, text="Play", command=self.play_song).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(controls_frame, text="Pause", command=self.pause_song).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(controls_frame, text="⏮ Prev", command=self.previous_song).grid(row=0, column=2, padx=10, pady=10)
        ttk.Button(controls_frame, text="Next ⏭", command=self.next_song).grid(row=0, column=3, padx=10, pady=10)

        ttk.Label(controls_frame, text="Volume:").grid(row=1, column=0, padx=10, pady=10)
        self.volume_slider = Scale(controls_frame, from_=0, to=100, orient="horizontal", length=200)
        self.volume_slider.set(50)
        self.volume_slider.grid(row=1, column=1, columnspan=3, padx=10, pady=10)

    # Placeholder methods for functionalities
    def start_microphone(self):
        messagebox.showinfo("Microphone", "Microphone started! Please speak.")
        command, speech = self.functions.start_microphone()
        self.text_label.config(text=command) 
        speech_array = self.functions.process_audio_file(speech)
        dimensional = self.functions.predict_valence_arousal(speech_array)
        print(dimensional)
        self.valence_slider.set(dimensional[0]*self.valence_slider.cget("to"))
        self.arousal_slider.set(dimensional[1]*self.valence_slider.cget("to"))
        playlist = self.functions.generate_playlist(dimensional)
        self.update_playlist(playlist)

    def update_playlist(self, playlist):
        self.playlist_box.delete(0, tk.END)
        self.current_playlist = playlist
        for idx, song in playlist.iterrows():
            self.playlist_box.insert(tk.END, f"{song['title']} - {song['artist']}")

    def select_song(self, event):
        selection = self.playlist_box.curselection()
        if selection:
            song = self.current_playlist.iloc[selection[0]]
            messagebox.showinfo("Playing", f"Playing {song['title']} by {song['artist']}")
    

    def generate_playlist_from_text(self):
        text_input = self.text_entry.get()
        if text_input:
            messagebox.showinfo("Generating Playlist", f"Generating playlist based on text input: {text_input}")
        else:
            messagebox.showwarning("Warning", "Please enter text to generate playlist.")

    def adjust_recommendation(self):
        valence = self.valence_slider.get()
        arousal = self.arousal_slider.get()
        playlist = self.functions.generate_playlist([valence, arousal])
        self.update_playlist(playlist)
        messagebox.showinfo("Adjusting Recommendation", f"Adjusting playlist with valence {valence} and arousal {arousal}")

    def play_song(self):
        messagebox.showinfo("Play", "Playing song")

    def pause_song(self):
        messagebox.showinfo("Pause", "Pausing song")

    def next_song(self):
        messagebox.showinfo("Next Song", "Skipping to next song")

    def previous_song(self):
        messagebox.showinfo("Previous Song", "Going to previous song")

if __name__ == "__main__":
    root = tk.Tk()
    root.option_add("*Font", "Arial 12")
    app = Recommersion(root)
    root.mainloop()
