import tkinter as tk
from tkinter import ttk, messagebox, Scale
from vocal_assistant.vocal_assistant import VocalAssistant
from vocal_assistant.emotion.emotion_model import process_func
from vocal_assistant.emotion.predict_emotion import load_trained_model, predict_emotion, return_device
from vocal_assistant.emotion.emotion_model import process_func
import pandas as pd
import numpy as np
import threading
import pickle
from pathlib import Path
import sounddevice as sd
import time
import pygame
import sys


class Functions:
    def __init__(self, callback):
        self.callback = callback
        self.vc = VocalAssistant(1)
        self.audeering_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.custom_model_name = "custom_model.pth"
        self.pretrained_model = "facebook/wav2vec2-base"
        threading.Thread(target=self.load_data).start()
        
    
    def get_model_from_name(self, name):
        d = {"Audeering": self.audeering_model_name,
             "Custom": self.custom_model_name}
        return d[name]


    def load_data(self):
        self.device = return_device()
        self.custom_model, self.custom_processor = load_trained_model(self.device, \
                                                                      self.custom_model_name, self.pretrained_model)
        
        self.custom = True if Path(self.custom_model_name).exists() else False
        print("self.custom,",self.custom)

        with open("./data/Songs_path", "rb") as f:
            self.data = pickle.load(f)
        print("Songs loaded!")

        self.callback()

    def start_microphone(self):
        self.vc.talk("What is your mood today?")
        return self.vc.take_vocal_command() 
    
    def process_audio_file(self, file):
        return self.vc.process_audio_file(file)
    
    def predict_valence_arousal(self, file):
        return predict_emotion(self.custom_model, self.device, self.custom_processor, file)[0].tolist() if self.custom else \
               process_func(file, 16000)[0][:2]
    
    def generate_playlist(self, dimensional, cut = 10):
        songs_list = pd.DataFrame({"id": self.data["musicId"], "eucl_dist":self.data[["Valence", "Arousal"]]\
                           .apply(lambda x: np.linalg.norm(x - dimensional), axis=1), "Valence": self.data["Valence"], \
                            "Arousal": self.data["Arousal"],\
                            "title":self.data["title"], "artist": self.data["artist"], "mp3_path":self.data["mp3_path"]})

        return songs_list.sort_values(by="eucl_dist")[:cut]


class Recommersion:
    def __init__(self, root):
        self.root = root
        self.root.title("Recommersion - Emotion-Based Music Recommendation")
        self.root.geometry("800x800")
        self.root.resizable(False, False)

        self.mixer = pygame.mixer
        self.mixer.init()
        self.mixer.music.set_volume(0.5)
        self.NEXT = pygame.USEREVENT + 1
        self.mixer.music.set_endevent(self.NEXT) 

        self.current_song = pd.DataFrame([])
        self.paused = True

        
        self.create_widgets()
        self.text_var = tk.StringVar()
        self.text_var.set("Loading dataset, please wait...")
        self.loading_label = ttk.Label(self.root, textvariable=self.text_var, font=("Arial", 16))
        self.loading_label.pack(pady=20)
        
        self.functions = None
        self.data_loaded = False
        self.initialize_functions()
        self.running = True
        self.event_thread = threading.Thread(target=self.handle_pygame_events, daemon=True)
        self.event_thread.start()


    def handle_pygame_events(self):
        while self.running:
            if (not self.mixer.music.get_busy()) and (self.mixer.music.get_endevent() == self.NEXT) and not self.paused:
                    self.next_song()
                    print("Next Song")
            time.sleep(0.5)  

    def initialize_functions(self):
        self.functions = Functions(self.on_data_loaded)

    def on_data_loaded(self):
        # Remove loading message
        self.text_var.set("Data Loaded: Songs dataset loaded successfully!")
        self.data_loaded = True
        time.sleep(10)
        self.loading_label.pack_forget()


    def create_widgets(self):
        self.create_input_frame()
        self.create_playlist_frame()
        self.create_adjustment_frame()
        self.create_controls_frame()


    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Input Emotion")
        input_frame.pack(pady=10, padx=10, fill="both", expand="yes")
        
        ttk.Button(input_frame, text="🎤 Speak Emotion", command=self.start_microphone).grid(row=0, column=0, padx=10, pady=10)
        
        self.text_label= ttk.Label(input_frame, text="Your speech text:", font=("Arial", 16), width=100)
        self.text_label.grid(row=0, column=1, padx=10, pady=10)

        #TODO: fix it
        OPTIONS = [
                    "Audeering",
                    "Custom"
                ] 
        self.model = tk.StringVar()
        self.model.set(OPTIONS[0])
        self.menu = ttk.OptionMenu(input_frame, self.model, *OPTIONS)
        self.menu.grid(row=0, column=1, padx=10, pady=10)
        

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
        
        self.playlist_label = tk.Listbox(playlist_frame, selectmode=tk.SINGLE, width=40, height=10)
        self.playlist_label.pack(padx=10, pady=10, fill="both", expand=True)
        self.playlist_label.bind('<<ListboxSelect>>', self.play_in_the_box)

    def create_controls_frame(self):
        controls_frame = ttk.LabelFrame(self.root, text="Playback Controls")
        controls_frame.pack(pady=10, padx=10, fill="both", expand="yes")

        ttk.Button(controls_frame, text="▷ Play", command=self.play_button).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(controls_frame, text="⏸️ Pause", command=self.pause_song).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(controls_frame, text="⏮ Prev", command=self.previous_song).grid(row=0, column=2, padx=10, pady=10)
        ttk.Button(controls_frame, text="Next ⏭", command=self.next_song).grid(row=0, column=3, padx=10, pady=10)

        ttk.Label(controls_frame, text="Volume:").grid(row=1, column=0, padx=10, pady=10)
        self.volume_slider = Scale(controls_frame, from_=0, to=100, orient="horizontal", length=200, command = self.manage_volume)
        self.volume_slider.set(50)
        self.volume_slider.grid(row=1, column=1, columnspan=3, padx=10, pady=10)


    # Placeholder methods for functionalities
    def start_microphone(self):
        if self.data_loaded:
            messagebox.showinfo("Microphone", "Microphone started! Please speak.")
            command, speech = self.functions.start_microphone()
            self.text_label.config(text=command) 
            speech_array = self.functions.process_audio_file(speech)
            dimensional = self.functions.predict_valence_arousal(speech_array)
            print(dimensional)
            print("before setting")
            self.root.after(0, lambda: self.valence_slider.set(dimensional[0] * self.valence_slider.cget("to")))
            self.root.after(0, lambda: self.arousal_slider.set(dimensional[1] * self.arousal_slider.cget("to")))
            print("after setting")
            playlist = self.functions.generate_playlist(dimensional)
            print(playlist)
            self.root.after(0, lambda: self.update_playlist(playlist))
            self.running = True
        else:
            messagebox.showinfo("Microphone", "No songs to compute yet")


    def update_playlist(self, playlist):
        #global current_song, paused
        self.playlist_label.delete(0, tk.END)  # Clear existing items
        self.current_playlist = playlist  # Save playlist for further use
        for idx, song in playlist.iterrows():
            self.playlist_label.insert(tk.END, f"{song['title']} - {song['artist']}")
        self.playlist_label.selection_set(0)
        self.current_song = self.current_playlist.iloc[self.playlist_label.curselection()[0]]
        self.paused = False
        self.play_song()

    

    def adjust_recommendation(self):
        if self.data_loaded:
            valence = self.valence_slider.get() / self.valence_slider.cget("to")
            arousal = self.arousal_slider.get() / self.valence_slider.cget("to")
            print(valence, arousal)
            playlist = self.functions.generate_playlist([valence, arousal])
            print(playlist)
            messagebox.showinfo("Adjusting Recommendation", f"Adjusting playlist with valence {valence} and arousal {arousal}")
            self.root.after(0, lambda :self.update_playlist(playlist))
            self.running = True
        else:
            messagebox.showinfo("Recompute playlist", "No songs to compute yet")


    def _play(self, song):
        self.mixer.music.load(song['mp3_path'])
        self.mixer.music.play(fade_ms=200)
    

    def play_in_the_box(self, event):
        return self.play_song()
    

    def play_button(self):
        if not (self.mixer.music.get_busy()):
            self.play_song()


    def play_song(self):
        selection = self.playlist_label.curselection()
        if selection and (not self.paused):
            self.current_song = self.current_playlist.iloc[selection[0]]
            self._play(self.current_song)
            
        else:
            self.mixer.music.unpause()
            self.paused = False


    def pause_song(self):
        self.mixer.music.pause()
        self.paused = True


    def next_song(self):
        selection = self.playlist_label.curselection()
        next = selection[0]+1
        if next < len(self.current_playlist):
            self.current_song = self.current_playlist.iloc[selection[0]+1]
            self.playlist_label.select_clear(0, tk.END)
            self.playlist_label.selection_set(next)
            self.paused = False
            self.play_song()
        else:
            self.running = False

        
    def previous_song(self):
        selection = self.playlist_label.curselection()
        prev = selection[0]-1
        if prev >= 0:
            self.current_song = self.current_playlist.iloc[selection[0]-1]
            self.playlist_label.select_clear(0, tk.END)
            self.playlist_label.selection_set(prev)
            self.paused = False
            self.play_song()
        else:
            self.running = False    

    def manage_volume(self, event):
        self.mixer.music.set_volume(self.volume_slider.get()/100)

    def fade_out_and_exit(self):
        print("Performing fade-out before exit...")
        root.stop_threads()
        self.mixer.music.fadeout(300)  
        self.time.wait(300)  
        self.mixer.quit()  


    def __del__(self):
        self.fade_out_and_exit()



if __name__ == "__main__":
    try:
        while True:
            root = tk.Tk()
            root.option_add("*Font", "Arial 12")
            app = Recommersion(root)
            root.mainloop()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting gracefully...")
        root.stop_threads()
        sys.exit(0)
    except SystemExit:
        print("SystemExit received. Performing cleanup...")
        root.stop_threads()
        sys.exit(0)


