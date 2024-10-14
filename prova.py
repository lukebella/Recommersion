from vocal_assistant import VocalAssistant
from emotion_model import process_func, EmotionModel
from transformers import Wav2Vec2Processor


device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

vc = VocalAssistant(1)
vc.talk("What is your mood today?")
while True:
    command, vocal_file = vc.take_command()
    print(command)
    break

print(process_func(vocal_file, 16000))