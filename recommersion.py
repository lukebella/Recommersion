import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, messagebox, Scale
from vocal_assistant.vocal_assistant import VocalAssistant
from vocal_assistant.emotion.emotion_model import process_func
from vocal_assistant.emotion.predict_emotion import load_trained_model, predict_emotion, return_device
from vocal_assistant.emotion.emotion_model import process_func
import pandas as pd
import numpy as np
import threading
import pickle
import time
import pygame
from scipy.spatial.distance import cosine


ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Functions:
    def __init__(self, callback):
        self.callback = callback
        self.vc = VocalAssistant(1)
        
        self.custom_model_name = "custom_model.pth"
        self.custom_pretrained_model = "facebook/wav2vec2-base"

        threading.Thread(target=self.load_data).start()

        
    def get_model_from_name(self, name):
        if name in self.d:
            return self.d[name]
        else:
            raise ValueError(f"Unknown model name: {name}")


    def load_data(self):
        self.device = return_device()
        self.custom_model, self.custom_processor = load_trained_model(self.device, \
                                                                      self.custom_model_name, self.custom_pretrained_model)
        
        with open("./data/Songs_path", "rb") as f:
            self.data = pickle.load(f)
        print("Songs loaded!")

        self.callback()

    def start_microphone(self):
        self.vc.talk("What is your mood today?")
        return self.vc.take_vocal_command() 
    
    def process_audio_file(self, file):
        return self.vc.process_audio_file(file)
    

    def predict_valence_arousal(self, file, model):
        print("Using model:", model)
        d = {
            "Audeering": process_func(file, 16000)[0][:2],
            "Custom": predict_emotion(self.custom_model, self.device, self.custom_processor, file)[0].tolist()
        }
        return d[model]
       
    
    def generate_playlist(self, dimensional, distance:str,  cut:int = 10):
        dist_dict = {
            "Euclidean": lambda x, y: np.linalg.norm(x-y),
            "Cosine": lambda x, y: cosine(x, y)
        }
        dist_func = dist_dict[distance]
        songs_list = pd.DataFrame({"id": self.data["musicId"], "dist":self.data[["Valence", "Arousal"]]\
                           .apply(lambda x: dist_func(x, dimensional), axis=1), "Valence": self.data["Valence"], \
                            "Arousal": self.data["Arousal"],\
                            "title":self.data["title"], "artist": self.data["artist"], "mp3_path":self.data["mp3_path"]})

        return songs_list.sort_values(by="dist")[:cut]


class Recommersion(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Recommersion - Emotion-Based Music Recommendation")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.mixer = pygame.mixer
        self.mixer.init()
        self.mixer.music.set_volume(0.5)
        self.NEXT = pygame.USEREVENT + 1
        self.mixer.music.set_endevent(self.NEXT) 

        self.current_song = pd.DataFrame([])
        self.paused = True

        
        self.text_var = tk.StringVar()
        self.text_var.set("Loading dataset, please wait...")
        self.loading_label = ctk.CTkLabel(self, textvariable=self.text_var, font = ctk.CTkFont(size=16, weight="bold"))
        self.loading_label.grid(pady=20)
        
        self.functions = None
        self.data_loaded = False
        self.initialize_functions()
        self.playlist_initialized = False
        self.poll_pygame_events()
            

        # create sidebar frame with widgets
        self.input_frame = ctk.CTkFrame(self, width=140, corner_radius=0, fg_color="black")
        self.input_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.input_frame.grid_rowconfigure(5, weight=1)
        self.input_frame.grid_columnconfigure(2, weight=1)

        self.logo_label = ctk.CTkLabel(self.input_frame, text="Input", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.microphone = ctk.CTkButton(self.input_frame, text="🎤 Speak Emotion", command=self.start_microphone)
        self.microphone.grid(row=1, column=0, padx=20, pady=10)
        self.text_label = ctk.CTkLabel(self.input_frame, text="Your speech text:", font=("Arial", 16), anchor="w")
        self.text_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.model_label = ctk.CTkLabel(self.input_frame, text="Model:", font=("Arial", 16), anchor="w")
        self.model_label.grid(row=2, column=0, padx=20, pady=10)

        self.MODEL_OPTIONS = ["Audeering", "Custom"]
        self.model = ctk.StringVar(value=self.MODEL_OPTIONS[0])  # Set the default selection
        self.menu = ctk.CTkOptionMenu(self.input_frame, values = self.MODEL_OPTIONS, command = self.on_option_change)
        self.menu.grid(row=2, column=1, padx=10, pady=10)
        self.model.trace_add("write", self.on_option_change)

        self.cut_label = ctk.CTkLabel(self.input_frame, text="Number of Songs:", font=("Arial", 16), anchor="w")
        self.cut_label.grid(row=3, column=0, padx=20, pady=10)

        self.cut_text = ctk.StringVar(value="10")  # Default value
        self.cut = ctk.CTkEntry(self.input_frame, placeholder_text=self.cut_text.get())
        self.cut.grid(row=3, column=1, padx=10, pady=10)
        self.cut_text.trace_add("write", self.on_text_change)

        self.distance_label = ctk.CTkLabel(self.input_frame, text="Distance", font=("Arial", 16), anchor="w")
        self.distance_label.grid(row=4, column=0, padx=20, pady=10)

        self.DISTANCE_OPTIONS = ["Euclidean", "Cosine"]
        self.distance = ctk.StringVar(value=self.DISTANCE_OPTIONS[0])  # Set the default selection
        self.menu_distance = ctk.CTkOptionMenu(self.input_frame, values = self.DISTANCE_OPTIONS, command = self.on_option_change_distance)
        self.menu_distance.grid(row=4, column=1, padx=10, pady=10)
        self.distance.trace_add("write", self.on_option_change_distance)

    
        self.after(2000, self.update_text)

        self.playlist_and_control_frame = ctk.CTkFrame(self, width = 1000)
        self.playlist_and_control_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.playlist_and_control_frame.grid_rowconfigure(2, weight=1)
        self.playlist_and_control_frame.grid_columnconfigure(0, weight=1)

        # create main entry and button
        self.controls_frame = ctk.CTkFrame(self, width = 1000)
        self.controls_frame.grid(row=2, column=1, columnspan=2, padx=(20, 20), pady=(20,0), sticky="nsew")
        self.controls_frame.grid_columnconfigure(4, weight=1)
        self.controls_frame.grid_rowconfigure(2, weight=1)

        ctk.CTkButton(self.controls_frame, text="⏮ Prev", command=self.previous_song).grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkButton(self.controls_frame, text="▷ Play", command=self.play_button).grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkButton(self.controls_frame, text="⏸️ Pause", command=self.pause_song).grid(row=0, column=2, padx=10, pady=10)
        ctk.CTkButton(self.controls_frame, text="Next ⏭", command=self.next_song).grid(row=0, column=3, padx=10, pady=10)

        ctk.CTkLabel(self.controls_frame, text="Volume:").grid(row=1, column=0, padx=10, pady=10)
        self.volume_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, orientation="horizontal", command=self.manage_volume)
        self.volume_slider.set(50)
        self.volume_slider.grid(row=1, column=2, columnspan=3, padx=10, pady=10)

        self.playlist_label = tk.Listbox(self.playlist_and_control_frame, selectmode=tk.SINGLE, width=40, height=10)
        self.playlist_label.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.playlist_label.bind('<<ListboxSelect>>', self.play_in_the_box)

        # create textbox
        self.textbox = ctk.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.textbox.grid_columnconfigure(0, weight=1)  

        # create slider and progressbar frame
        self.adjustment_frame = ctk.CTkFrame(self.playlist_and_control_frame, fg_color="transparent")
        self.adjustment_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.adjustment_frame.grid_columnconfigure(1, weight=1)
        self.adjustment_frame.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(self.adjustment_frame, text="Valence:").grid(row=0, column=0, padx=10, pady=10)
        self.valence_slider = ctk.CTkSlider(self.adjustment_frame, from_=0, to=1000, orientation="horizontal")
        self.valence_slider.grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(self.adjustment_frame, text="Arousal:").grid(row=1, column=0, padx=10, pady=10)
        self.arousal_slider = ctk.CTkSlider(self.adjustment_frame, from_=0, to=1000, orientation="horizontal")
        self.arousal_slider.grid(row=1, column=1, padx=10, pady=10)

        ctk.CTkButton(self.adjustment_frame, text="Recompute Playlist", command=self.adjust_recommendation).grid(row=2, column=0, columnspan=2, pady=10)
        
    def poll_pygame_events(self):
        if self.playlist_initialized:
            if not (self.mixer.music.get_busy() or self.paused):
                self.next_song()
        self.after(150, self.poll_pygame_events)


    def initialize_functions(self):
        self.functions = Functions(self.on_data_loaded)

    def on_data_loaded(self):
        # Remove loading message
        self.text_var.set("Data Loaded: Songs dataset loaded successfully!")
        self.data_loaded = True
        time.sleep(10)
        self.loading_label.pack_forget()
  
    
    def on_text_change(self, *args):
        print(f"Current text in Entry: {self.cut_text.get()}")

    def update_text(self):
        self.cut_text.set(self.cut_text.get())


    def on_option_change(self, *args):
        print(f"Selected model option: {self.model.get()}")
    
    def on_option_change_distance(self, *args):
        print(f"Selected distance option: {self.distance.get()}")
    
    def check_cut(self, cut:str):
        if cut.isdigit() and int(cut)>0 and int(cut)<len(self.functions.data):
            return int(cut)
        else:
            messagebox.showinfo("Value Error", "Set a positive integer type greater than zero or a minor number!")
            return None
    

    def compute_playlist(self, dimensional):
        cut = self.check_cut(self.cut_text.get())
        if cut is None:
            return
        playlist = self.functions.generate_playlist(dimensional = dimensional, distance=self.distance.get(), cut = cut)
        messagebox.showinfo("Adjusting Recommendation", \
                            f"Adjusting playlist with valence {dimensional[0]} and arousal {dimensional[1]}")
        print(playlist)
        self.after(0, lambda: self.update_playlist(playlist))


    # Placeholder methods for functionalities
    def start_microphone(self):
        if self.data_loaded:
            messagebox.showinfo("Microphone", "Microphone started! Please speak.")
            self.pause_song()
            command, speech = self.functions.start_microphone()
            self.text_label.config(text=command) 
            speech_array = self.functions.process_audio_file(speech)
            dimensional = self.functions.predict_valence_arousal(file = speech_array, \
                                                                 model = self.model.get())
            print(dimensional)
            print("before setting")
            self.after(0, lambda: self.valence_slider.set(dimensional[0] * self.valence_slider.cget("to")))
            self.after(0, lambda: self.arousal_slider.set(dimensional[1] * self.arousal_slider.cget("to")))
            print("after setting")
            self.compute_playlist(dimensional)
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
        self.playlist_initialized = True  # Set the flag here
        self.play_song()

    
    def adjust_recommendation(self):
        if self.data_loaded:
            valence = self.valence_slider.get() / self.valence_slider.cget("to")
            arousal = self.arousal_slider.get() / self.valence_slider.cget("to")
            print(valence, arousal)
            self.compute_playlist([valence, arousal])
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
            self.playlist_initialized = False

        
    def previous_song(self):
        selection = self.playlist_label.curselection()
        prev = selection[0]-1
        if prev >= 0:
            self.current_song = self.current_playlist.iloc[selection[0]-1]
            self.playlist_label.select_clear(0, tk.END)
            self.playlist_label.selection_set(prev)
            self.paused = False
            self.play_song()


    def manage_volume(self, event):
        self.mixer.music.set_volume(self.volume_slider.get()/100)

    def fade_out_and_exit(self):
        print("Performing fade-out before exit...")
        #root.stop_threads()
        self.mixer.music.fadeout(300)  
        self.time.wait(300)  
        self.mixer.quit()  


    def __del__(self):
        self.fade_out_and_exit()
        

if __name__ == "__main__":
    app = Recommersion()
    app.mainloop()



