import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, ttk
from vocal_assistant.vocal_assistant import VocalAssistant
from vocal_assistant.emotion.emotion_model import process_func
from vocal_assistant.emotion.predict_emotion import load_trained_model, predict_emotion, return_device
from vocal_assistant.emotion.emotion_model import process_func
import pandas as pd
import numpy as np
import threading
import pickle
import pygame
from scipy.spatial.distance import cosine
from hover_interface.hover_text import HoverText
from PIL import Image

# Appearence and default color
ctk.set_appearance_mode("System")  
#ctk.set_default_color_theme("dark-blue")  


# Functions used by the GUI
class Functions:
    def __init__(self, callback):
        self.callback = callback
        self.vc = VocalAssistant(1)
        
        self.custom_model_name = "custom_model.pth"
        self.custom_pretrained_model = "facebook/wav2vec2-base"

        threading.Thread(target=self.load_data).start()

    # Loading data parallely
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
    
    # Predicting dimensional values via the model name
    def predict_valence_arousal(self, file, model):
        print("Using model:", model)
        d = {
            "Audeering": process_func(file, 16000)[0][:2],
            "Custom": predict_emotion(self.custom_model, self.device, self.custom_processor, file)[0].tolist()
        }
        return d[model]
       
    
    # Generating playlist via the selected distance
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

        self.title("Recommersion - Emotion-Based Music Recommendation")
        self.geometry(f"{1550}x{950}")
        self.resizable(True, True)
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        # Tool for managing songs reproduction
        self.mixer = pygame.mixer
        self.mixer.init()
        self.mixer.music.set_volume(0.5)
        self.NEXT = pygame.USEREVENT + 1
        self.mixer.music.set_endevent(self.NEXT) 

        self.current_song = pd.DataFrame([])
        self.paused = True
        
        # Class for display text while hovering any widget 
        self.hover_text = HoverText("hover_interface/widget_hover.txt")
        
        self.text_var = ctk.StringVar()
        self.text_var.set("Loading dataset, please wait...")
        self.loading_label = ctk.CTkLabel(self, text= self.text_var.get(), textvariable=self.text_var, font = ctk.CTkFont(size=16, weight="bold"))
        self.loading_label.grid(row=4, column=0, sticky="nsew")
        
        self.functions = None
        self.data_loaded = False
        self.initialize_functions()
        self.playlist_initialized = False
        self.poll_pygame_events()
        
        # Frame and widget creations
        self.create_input_frame()
        self.create_playlist_frame()
        
        
    def create_input_frame(self):
        self.input_frame = ctk.CTkFrame(self, width=140, corner_radius=0, fg_color="gray5", bg_color="white")
        self.input_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.input_frame.grid_rowconfigure(9, weight=1)
        self.input_frame.grid_columnconfigure(2, weight=1)

        # Voice Input
        self.logo_label = ctk.CTkLabel(self.input_frame, text="Voice Input", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=10, pady=(20, 10))

        self.microphone = ctk.CTkButton(self.input_frame, text="🎤 Speak Emotion", command=self.start_microphone, \
                                        fg_color="SlateBlue4")
        self.microphone.grid(row=1, column=1, padx=(10,20), pady=10, sticky="nsew")
        self.microphone.bind("<Enter>", lambda event: self.hover_show(event, "microphone"))
        self.speech_var = ctk.StringVar()
        self.speech_var.set("Your speech text:")
        self.text_label = ctk.CTkLabel(self.input_frame, text=self.speech_var.get(), textvariable=self.speech_var, font=("Italic", 16), anchor="w")
        self.text_label.bind("<Enter>", lambda event: self.hover_show(event, "text_label"))
        self.text_label.grid(row=1, column=0, padx=10, pady=10)

        # Model input
        self.model_label = ctk.CTkLabel(self.input_frame, text="Model:", font=("Arial", 16), anchor="w")
        self.model_label.grid(row=2, column=0, padx=10, pady=10)

        self.MODEL_OPTIONS = ["Audeering", "Custom"]
        self.model = ctk.StringVar(value=self.MODEL_OPTIONS[0]) 
        self.menu = ctk.CTkOptionMenu(self.input_frame, values = self.MODEL_OPTIONS, \
                                      variable = self.model, command = self.on_option_change, \
                                      fg_color="SlateBlue4", dropdown_fg_color="SlateBlue4")
        self.menu.grid(row=2, column=1, padx=(10,20), pady=10, sticky="nsew")
        self.menu.bind("<Enter>", lambda event: self.hover_show(event, "menu"))
        self.model.trace_add("write", self.on_option_change)

        # Parameters Input
        self.parameters_label = ctk.CTkLabel(self.input_frame, text="Parameters", font=ctk.CTkFont(size=20, weight="bold"))
        self.parameters_label.grid(row=3, column=0, padx=10, pady=(61, 10))
        self.cut_label = ctk.CTkLabel(self.input_frame, text="Number of Songs:", font=("Arial", 16), anchor="w")
        self.cut_label.grid(row=4, column=0, padx=10, pady=10)

        self.cut_text = ctk.StringVar(value="10")  
        self.cut = ctk.CTkEntry(self.input_frame, placeholder_text=self.cut_text.get(), textvariable=self.cut_text)
        self.cut.grid(row=4, column=1, padx=(10,20), pady=10, sticky="nsew")
        self.cut.bind("<Enter>", lambda event: self.hover_show(event, "cut"))
        self.cut_text.trace_add("write", self.on_text_change)

        self.distance_label = ctk.CTkLabel(self.input_frame, text="Distance", font=("Arial", 16), anchor="w")
        self.distance_label.grid(row=5, column=0, padx=10, pady=10)

        self.DISTANCE_OPTIONS = ["Euclidean", "Cosine"]
        self.distance = ctk.StringVar(value=self.DISTANCE_OPTIONS[0])  
        self.menu_distance = ctk.CTkOptionMenu(self.input_frame, values = self.DISTANCE_OPTIONS, \
                                               variable=self.distance, command = self.on_option_change_distance, \
                                               fg_color="SlateBlue4", dropdown_fg_color="SlateBlue4")
        self.menu_distance.grid(row=5, column=1, padx=(10,20), pady=10, sticky="nsew")
        self.menu_distance.bind("<Enter>", lambda event: self.hover_show(event, "menu_distance"))
        self.distance.trace_add("write", self.on_option_change_distance)

        # Instruction Textbox
        self.instr_label = ctk.CTkLabel(self.input_frame, text = "Instructions",font=ctk.CTkFont(size=20, weight="bold"))
        self.instr_label.grid(row=7, column=0, padx=(20, 20), pady=(120, 10), sticky="w")
        self.textbox = ctk.CTkTextbox(self.input_frame, wrap = "word", height=220, width=200, font=ctk.CTkFont("italic", size=15))
        self.textbox.insert(1.0, self.hover_text.get_widget("general"))
        self.textbox.grid(row=8, column=0, padx=(20, 20), pady=(20, 10), sticky="nsew")

        # Quadrant frame
        self.quadrant_label = ctk.CTkLabel(self.input_frame, text = "Emotion Quadrants",font=ctk.CTkFont(size=20, weight="bold"))
        self.quadrant_label.grid(row=7, column=1, padx=(20, 20), pady=(120, 10), sticky="w")
        self.quadrant_frame = ctk.CTkFrame(self.input_frame, height=500, width=340, fg_color="white")
        self.quadrant_frame.grid(row=8, column=1, padx=(5, 20), pady=(10, 10), sticky="w")
        self.quadrant_frame.bind("<Enter>", lambda event: self.hover_show(event, "quadrant_frame"))
        self.image = ctk.CTkImage(light_image = Image.open("hover_interface/emotion_quadrant.drawio.png"),\
                                  dark_image = Image.open("hover_interface/emotion_quadrant.drawio.png"),
                                  size=(500,340)) 
        self.label_image = ctk.CTkLabel(self.quadrant_frame, image=self.image, text="")
        self.label_image.grid(row=0, column=0, padx=(5, 5), pady=(5, 5))
        self.after(2000, self.update_text)


    def create_playlist_frame(self):
        self.playlist_and_control_frame = ctk.CTkFrame(self, width = 500, bg_color ="white", fg_color="gray5", corner_radius=0)
        self.playlist_and_control_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.playlist_and_control_frame.grid_rowconfigure(6, weight=1)
        self.playlist_and_control_frame.grid_columnconfigure(1, weight=1)

        # Playlist box
        self.playlist_label = ctk.CTkLabel(self.playlist_and_control_frame, text="Playlist", \
                                           font=ctk.CTkFont(size=20, weight="bold"), text_color="white")
        self.playlist_label.grid(row=0, column=1, padx=20, pady=(18, 4))
        self.playlist_frame = ctk.CTkScrollableFrame(self.playlist_and_control_frame, fg_color="white", width=700, height =260)
        self.playlist_frame.grid(row=1, column=1, padx=(20, 20), pady=(20, 20), sticky="w")
        self.playlist_frame.grid_rowconfigure(0, weight=1)
        self.playlist_box = ttk.Treeview(self.playlist_frame, selectmode=tk.BROWSE, height=15)
        self.playlist_box.grid(row=0, column=0, padx=(1, 1), pady=(1, 1), sticky="ew")
        self.playlist_box["columns"] = ("Artist", "Song")
        self.playlist_box.column("#0", width=0, minwidth=0)
        self.playlist_box.column("Artist", width = 300, anchor=tk.W)
        self.playlist_box.column("Song", width = 400, anchor= tk.W)

        self.playlist_box.heading("Artist", text="Artist", anchor = tk.CENTER)
        self.playlist_box.heading("Song", text="Song", anchor = tk.CENTER)
        scrollbar = tk.Scrollbar(self.playlist_and_control_frame, orient=tk.VERTICAL, command=self.playlist_box.yview)
        self.playlist_box.config(yscrollcommand=scrollbar.set)
        self.canvas = ctk.CTkCanvas(self.playlist_box, width=2, height=200, bg="white", borderwidth=0, highlightthickness=0)

        self.playlist_box.bind('<<TreeviewSelect>>', self.play_in_the_box)
        self.playlist_box.bind("<Enter>", lambda event: self.hover_show(event, "playlist_box"))

        # Dimensional Sliders
        self.adj_label = ctk.CTkLabel(self.playlist_and_control_frame, text="Emotional Adjustments", \
                                      font=ctk.CTkFont(size=20, weight="bold"), text_color="white")
        self.adj_label.grid(row=2, column=1, padx=20, pady=(25, 3), sticky = "w")
        self.adjustment_frame = ctk.CTkFrame(self.playlist_and_control_frame, fg_color="transparent")
        self.adjustment_frame.grid(row=3, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.adjustment_frame.grid_columnconfigure(0, weight=0)  # Label column
        self.adjustment_frame.grid_columnconfigure(1, weight=1)  # Slider column
        self.adjustment_frame.grid_columnconfigure(2, weight=0)  # Emoji column
        self.adjustment_frame.grid_columnconfigure(3, weight=0)  # Indicator column
        self.adjustment_frame.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(self.adjustment_frame, text="Valence: 😞", text_color="white",font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=0, column=0, padx=0, pady=10)                                                                                         
        self.valence_slider = ctk.CTkSlider(self.adjustment_frame, from_=0, to=1000, orientation="horizontal", width = 500, command=self.update_valence_value)
        self.valence_slider.grid(row=0, column=1, padx=0, pady=5, sticky = 'ew')
        self.valence_slider.bind("<Enter>", lambda event: self.hover_show(event, "valence_slider"))
        ctk.CTkLabel(self.adjustment_frame, text="😁", text_color="white", font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=0, column=2, padx=(10,20), pady=5, sticky = 'nsew')
        self.valence_value = tk.StringVar(value=f"Value: {self.valence_slider.get()/1000:.4f}")
        ctk.CTkLabel(self.adjustment_frame, textvariable=self.valence_value, text_color="white", font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=0, column=3, padx=(10,20), pady=5, sticky = 'nsew')
        

        ctk.CTkLabel(self.adjustment_frame, text="Arousal: 🧘🏼", text_color="white", font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=1, column=0, padx=10, pady=10)
        self.arousal_slider = ctk.CTkSlider(self.adjustment_frame, from_=0, to=1000, orientation="horizontal", width = 500, command=self.update_arousal_value)
        self.arousal_slider.grid(row=1, column=1, padx=0, pady=10, sticky = 'ew')
        self.arousal_slider.bind("<Enter>", lambda event: self.hover_show(event, "arousal_slider"))
        ctk.CTkLabel(self.adjustment_frame, text="🔥", text_color="white", font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=1, column=2, padx=(10,20), pady=10, sticky='nsew')
        self.arousal_value = tk.StringVar(value=f"Value: {self.arousal_slider.get()/1000:.4f}")
        ctk.CTkLabel(self.adjustment_frame, textvariable=self.arousal_value, text_color="white", font=ctk.CTkFont(size=15, weight="bold"))\
                    .grid(row=1, column=3, padx=(10,20), pady=10, sticky = 'nsew')
        

        self.recompute_playlist = ctk.CTkButton(self.adjustment_frame, text="Recompute Playlist", command=self.adjust_recommendation, \
                                                width = 200, fg_color="SlateBlue4")
        
        self.recompute_playlist.grid(row=2, column=0, columnspan=2, padx=(200,20), pady=(40,30), sticky = "nsew")
        self.recompute_playlist.bind("<Enter>", lambda event: self.hover_show(event, "recompute_playlist"))
        
        # Song and Control frames
        self.song_frame = ctk.CTkFrame(self.playlist_and_control_frame, width = 500, height=50, fg_color="SlateBlue3")
        self.song_frame.grid(row=5, column=1, columnspan=4, padx=(20, 20), pady=(45,0), sticky="nsew")
        self.song_frame.bind("<Enter>", lambda event: self.hover_show(event, "song_frame"))
        self.controls_frame = ctk.CTkFrame(self.playlist_and_control_frame, width = 500, height=60, fg_color="SlateBlue4")
        self.controls_frame.grid(row=6, column=1, columnspan=4, padx=(20, 20), pady=(0,40), sticky="nsew")
        self.controls_frame.bind("<Enter>", lambda event: self.hover_show(event, "controls_frame"))
        self.controls_frame.grid_columnconfigure(6, weight=1)
        self.controls_frame.grid_rowconfigure(1, weight=1)


        self.song_var = ctk.StringVar()
        self.song_var.set("No song played...")
        self.song_label = ctk.CTkLabel(self.song_frame, text=self.song_var.get(), textvariable=self.song_var, width=500, \
                                       font=("Helvetica", 16, "bold"), anchor="center", text_color="black")
        self.song_label.grid(row = 0, column = 0, padx=(90, 100), pady=(20,20), sticky="nsew")
        ctk.CTkButton(self.controls_frame, width = 50, text="⏮", command=self.previous_song).grid(row=1, column=0, padx=(220,5), pady=10, sticky = "w")
        ctk.CTkButton(self.controls_frame, width = 50, text="▷", command=self.play_button).grid(row=1, column=1, padx=5, pady=10, sticky = "w")
        ctk.CTkButton(self.controls_frame, width = 50, text="⏸︎", command=self.pause_song).grid(row=1, column=2, padx=5, pady=10, sticky = "w")
        ctk.CTkButton(self.controls_frame, width = 50, text="⏭", command=self.next_song).grid(row=1, column=3, padx=(5,20), pady=10, sticky = "w")

        ctk.CTkLabel(self.controls_frame, text="Volume:").grid(row=1, column=4, padx=(60,10), pady=10)
        self.volume_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, orientation="horizontal", command=self.manage_volume)
        self.volume_slider.set(50)
        self.volume_slider.grid(row=1, column=5, columnspan=3, padx=10, pady=10, sticky = 'w')



    # Method for checking whether there are any other songs to be played automatically
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
        #self.after(10000, self.remove_loading_label())
    
    # def remove_loading_label(self):
    #     self.loading_label.pack_forget()

    def hover_show(self, event, s):
        text = self.hover_text.get_widget(s)
        self.textbox.delete(1.0, ctk.END)  
        self.textbox.insert(ctk.END, text)
    
    def update_valence_value(self, value):
        # Update the Valence StringVar with the slider's value
        self.valence_value.set(f"Value: {float(value)/1000:.4f}")  
    
    def update_arousal_value(self, value):
        # Update the Arousal StringVar with the slider's value
        self.arousal_value.set(f"Value: {float(value)/1000:.4f}")  
    
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
        playlist = self.functions.generate_playlist(dimensional = dimensional, \
                                                    distance=self.distance.get(), cut = cut).reset_index()
        messagebox.showinfo("Adjusting Recommendation", \
                            f"Adjusting playlist with valence {dimensional[0]} and arousal {dimensional[1]}")
        print(playlist)
        self.after(0, lambda: self.update_playlist(playlist))

    
    # Trigger when Microphone button is pressed
    def start_microphone(self):
        if self.data_loaded:
            messagebox.showinfo("Microphone", "Microphone started! Please speak.")
            self.pause_song()
            command, speech = self.functions.start_microphone()
            self.speech_var.set(value = command) 
            speech_array = self.functions.process_audio_file(speech)
            dimensional = self.functions.predict_valence_arousal(file = speech_array, \
                                                                 model = self.model.get())
            print(dimensional)
            print("before setting")
            new_val = dimensional[0] * self.valence_slider.cget("to")
            new_ar = dimensional[1] * self.arousal_slider.cget("to")
            self.after(0, lambda: self.valence_slider.set(new_val))
            self.after(0, lambda: self.arousal_slider.set(new_ar))
            self.after(0, lambda: self.update_valence_value(new_val))
            self.after(0, lambda: self.update_arousal_value(new_ar))
            print("after setting")
            self.compute_playlist(dimensional)
        else:
            messagebox.showinfo("Microphone", "No songs to compute yet")


    # Playlist creation or updating
    def update_playlist(self, playlist):
        
        self.playlist_box.delete(*self.playlist_box.get_children())  
        self.current_playlist = playlist  
        for i, song in playlist.iterrows():
            self.playlist_box.insert("", "end", iid=i, values=(song['artist'], song['title']))
            y = i * 20
            self.canvas.create_line(0, y, self.canvas.winfo_width(), y, fill="gray", width=1)
        first_item = int(self.playlist_box.get_children()[0])        
        self.playlist_box.selection_set(first_item)
        self.current_song = self.current_playlist.iloc[first_item]
        print(self.current_song)
        self.paused = False
        self.playlist_initialized = True  
        self.play_song()

    
    def adjust_recommendation(self):
        if self.data_loaded:
            valence = self.valence_slider.get() / self.valence_slider.cget("to")
            arousal = self.arousal_slider.get() / self.valence_slider.cget("to")
            print(valence, arousal)
            self.compute_playlist([valence, arousal])
        else:
            messagebox.showinfo("Recompute playlist", "No songs to compute yet")


    # Song control's methods

    def _play(self, song):
        self.mixer.music.load(song['mp3_path'])
        self.mixer.music.play(fade_ms=200)
    

    def play_in_the_box(self, event):
        return self.play_song()
    

    def play_button(self):
        if not (self.mixer.music.get_busy()):
            self.play_song()


    def play_song(self):
        selection = self.playlist_box.selection()
        idx = int(selection[0])
        if selection and (not self.paused):
            self.current_song = self.current_playlist.iloc[idx]
            self.song_var.set(f"{self.current_song['artist']} - {self.current_song['title']}")
            self._play(self.current_song)
            
        else:
            self.mixer.music.unpause()
            self.paused = False


    def pause_song(self):
        self.mixer.music.pause()
        self.paused = True


    def next_song(self):
        selection = self.playlist_box.selection()
        next = int(selection[0])+1
        if next < len(self.current_playlist):
            self.current_song = self.current_playlist.iloc[next]
            self.playlist_box.selection_set(next)
            self.paused = False
            self.play_song()
        else:
            self.playlist_initialized = False

        
    def previous_song(self):
        selection = self.playlist_box.selection()
        prev = int(selection[0])-1
        if prev >= 0:
            self.current_song = self.current_playlist.iloc[prev]
            self.playlist_box.selection_set(prev)
            self.paused = False
            self.play_song()


    def manage_volume(self, event):
        self.mixer.music.set_volume(self.volume_slider.get()/100)

    def fade_out_and_exit(self):
        print("Performing fade-out before exit...")
        self.mixer.music.fadeout(300)  
        self.time.wait(300)  
        self.mixer.quit()  


    def __del__(self):
        self.fade_out_and_exit()
        
# Main
if __name__ == "__main__":
    app = Recommersion()
    app.mainloop()



#Tests:
    #Try 3 emotion status
    #For Precision
    #Cut = 5
        #Custom/Euclidean
        #Custom/Cosine
        #Audeering/Euclidean
        #Audeering/Cosine
    #Cut = 10
        #Custom/Euclidean
        #Custom/Cosine
        #Audeering/Euclidean
        #Audeering/Cosine
    #For every test, check how much it is deviated with the sliders until the user finds the song
    #appropriate for his/her mood
    #General questions:
        #Do you find Recommersion useful in a music stream platform?
        #If yes, would you change something?