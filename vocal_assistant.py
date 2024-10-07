import speech_recognition as sr
import pyttsx3


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
                command = self.listener.recognize_google(voice)
                command = command.lower()
                """if 'alexa' in command:
                    command = command.replace('alexa','')
                    print(command)"""
        except:
            pass
        return command
    

if __name__ == '__main__':
    vc = VocalAssistant(1)
    vc.talk("What is your mood today?")
    while True:
        command = vc.take_command()
        print(command)
        exit()
    