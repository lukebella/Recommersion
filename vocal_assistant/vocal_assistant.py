import speech_recognition as sr
import pyttsx3
import librosa
import io
import os

class VocalAssistant:

    listener = sr.Recognizer()  

    def __init__(self, voice_type) -> None:
        # Check if running in an SSH environment (headless)
        if "SSH_CONNECTION" in os.environ:
            print("Detected SSH environment; skipping pyttsx3 initialization.")
            self.speak = None  # Skip pyttsx3 initialization
        else:
            self.speak = pyttsx3.init()  # Initialize pyttsx3 for non-SSH
            self.voices = self.speak.getProperty('voices')  # Now we can access voices after initialization
            self.speak.setProperty('voice', self.voices[voice_type].id)  # Set the desired voice

    def talk(self, audio) -> None:
        # If running in SSH, skip the talk method or simulate it with print
        if self.speak:
            self.speak.say(audio)
            self.speak.runAndWait()
        else:
            print(f"Simulating speech: {audio}")  # Placeholder for SSH environment

    def take_vocal_command(self) -> str:
        command = ""  # Initialize command with an empty string
        try:
            # Use the microphone
            with sr.Microphone() as source:  
                print("Listening....")
            
                # Use microphone as source and calling speech recognition to listen
                voice = self.listener.listen(source)
                wav_file = voice.get_wav_data()
                command = self.listener.recognize_google(voice)
                command = f"\"{command.lower()}\""
                print(command)
        except Exception as e:
            print(f"Error occurred: {e}")
        return command, wav_file
    

    def process_audio_file_str(self, file_path: str):
        # Initialize the recognizer
        command = ""
        audio = None
        try:
            # Load the audio file
            with sr.AudioFile(file_path) as source:
                print(f"Processing the file: {file_path}")
                audio_data = self.listener.record(source)  
                wav_file = audio_data.get_wav_data()
                audio, _ = librosa.load(io.BytesIO(wav_file), sr=16000)
                print("Numpy array shape:", audio.shape)
                # Recognize the speech in the audio file with Google Speech API
                command = self.listener.recognize_google(audio_data) 
                print(f"Recognized speech: {command.lower()}")
        except Exception as e:
            print(f"Error occurred: {e}")
        
        return audio
    
    def process_audio_file(self, wav_file:bytes):
        # Initialize the recognizer
        audio, _ = librosa.load(io.BytesIO(wav_file), sr=16000)
        return audio
    
        


if __name__ == '__main__':
    vc = VocalAssistant(1)
    vc.talk("What is your mood today?")  # This will be skipped if running over SSH
    file_path = "happy.wav" 
    audio = vc.process_audio_file_str(file_path)
    exit()
