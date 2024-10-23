import speech_recognition as sr
import pyttsx3
import librosa
import io


class VocalAssistant:

    listener = sr.Recognizer()  
    speak = pyttsx3.init()  
    voices = speak.getProperty('voices')
    
    def __init__(self, voice_type) -> None:
        self.speak.setProperty('voice', self.voices[voice_type].id)

    def talk(self, audio) -> None:
        self.speak.say(audio)
        self.speak.runAndWait()
    
    def take_command(self) -> str:
        command = ""  # Initialize command with an empty string
        try:
            with sr.Microphone() as source: # use the microphone
                print("  listening....")
            
                #use our microphone as source and calling speechrecognizier to listen this source
                voice = self.listener.listen(source)
                wav_file = voice.get_wav_data()

                audio, sample_rate = librosa.load(io.BytesIO(wav_file), sr=16000)
    
                # Print sample rate and the shape of the numpy array
                print("Sample rate:", sample_rate)
                print("Numpy array shape:", audio.shape)
                command = self.listener.recognize_google(voice)
                command = command.lower()
               
        except:
            pass
        return command, audio
    

    

if __name__ == '__main__':
    vc = VocalAssistant(1)
    vc.talk("What is your mood today?")
    while True:
        command, vocal_file = vc.take_command()
        print(command)
        exit()
    