import pandas as pd
import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Config
import random
from sklearn.model_selection import train_test_split
from predict_emotion import return_device, AudioAugmentation, L1, L2, R2, ccc_loss, predict_emotion, EmotionModel, load_trained_model
from emotion_model import process_func

device = "cpu"  #return_device()
#model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
model_name = "facebook/wav2vec2-base"
#processor = Wav2Vec2Processor.from_pretrained(model_name)
config = Wav2Vec2Config.from_pretrained(model_name)
model, processor = load_trained_model(device,"custom_2.pth", model_name)

muse = pd.read_pickle("data/MuSe_sample").sample(frac=1, random_state=42)
iemocap = pd.read_pickle("data/IEMOCAP_useful").sample(frac=1, random_state=42)
msp = pd.read_pickle("data/MSP_PODCAST_SAMPLED").sample(frac=1, random_state=42)
df = pd.concat([iemocap, muse, msp]).sample(frac=1, random_state=42)

df.drop(columns = ["Name"], inplace = True)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

augmenter = AudioAugmentation(sample_rate=16000)

print(test_df)
max_length = 16000 * 6

logits = None

filtered_df = test_df.loc[test_df["wav_file"].apply(len) <= max_length /0.8]

for i in range(len(filtered_df)):
    cur_wav = filtered_df.iloc[i]["wav_file"]
    random_wav = filtered_df.sample(n=1).iloc[0]["wav_file"]
    
    rand_augmenter = int(random.random() * 1000)
    random_wav = (torch.randn_like(torch.from_numpy(cur_wav)) * 0.01).numpy() if len(random_wav) < len(cur_wav) \
            else random_wav[:len(cur_wav)]

    wav_data = augmenter.augment(cur_wav, random_wav) if (rand_augmenter % 4 == 0) else cur_wav

    #log = torch.from_numpy(process_func(wav_data, 16000)[0][:2]).to(device).unsqueeze(0)
    log = predict_emotion(model, device, processor, wav_data)
    print(log)
    logits = torch.cat((logits, log)) if logits is not None else log
    print(logits[:,0])
    print(f"{(len(logits)/len(filtered_df))*100:.3f}%")

 
valence_tensor = torch.tensor(filtered_df["Valence"].values, dtype=torch.float32, device=device)
arousal_tensor = torch.tensor(filtered_df["Arousal"].values, dtype=torch.float32, device=device)


print("CCC Valence:",ccc_loss(valence_tensor, logits[:,0]).item())
print("L1 Valence:" ,L1(valence_tensor, logits[:,0]).item())
print("L2 Valence:" ,L2(valence_tensor, logits[:,0]).item())
print("R2 Valence:" ,R2(valence_tensor, logits[:,0]).item())
print()
print("CCC Arousal:",ccc_loss(arousal_tensor, logits[:,1]).item())
print("L1 Arousal:" ,L1(arousal_tensor, logits[:,1]).item()) 
print("L2 Arousal:" ,L2(arousal_tensor, logits[:,1]).item())
print("R2 Arousal:" ,R2(arousal_tensor, logits[:,1]).item())


# **********Audeering**********
# CCC Valence: 0.45900028944015503
# L1 Valence: 0.13037016987800598
# L2 Valence: 0.029216410592198372
# R2 Valence: 0.06784600019454956

# CCC Arousal: 0.4048195481300354
# L1 Arousal: 0.11777794361114502
# L2 Arousal: 0.023512154817581177
# R2 Arousal: 0.15354233980178833


# **********Custom**********
# CCC Valence: 0.7543468475341797
# L1 Valence: 0.14536753296852112
# L2 Valence: 0.0337047353386879
# R2 Valence: -0.07535457611083984

# CCC Arousal: 0.6669815182685852
# L1 Arousal: 0.18179208040237427
# L2 Arousal: 0.04841628670692444
# R2 Arousal: -0.743027925491333