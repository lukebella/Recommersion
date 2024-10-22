import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from vocal_assistant import VocalAssistant
from emotion_model import process_func, EmotionModel
from transformers import Wav2Vec2Processor
from predict_emotion import load_trained_model, predict_emotion

device = 'cpu'

audeering_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
audeering_processor = Wav2Vec2Processor.from_pretrained(audeering_model_name)
audeering_model = EmotionModel.from_pretrained(audeering_model_name).to(device)

custom_model_name = "model_checkpoint.pth"
custom_model, custom_processor = load_trained_model(custom_model_name)


vc = VocalAssistant(1)
vc.talk("What is your mood today?")
while True:
    command, vocal_file = vc.take_command()
    print(command)
    break

print("Audeering: ")
print(process_func(vocal_file, 16000))
print()
print("Custom: ")
print(predict_emotion(custom_model, custom_processor, vocal_file))