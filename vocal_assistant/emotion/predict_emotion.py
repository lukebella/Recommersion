import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Config
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchmetrics.regression import ConcordanceCorrCoef
import numpy as np
import random
import librosa
from pathlib import Path
import gc



# Class for implementing audio augmentations
class AudioAugmentation:
    def __init__(self, sample_rate=16000, noise_level=0.01, time_mask_param=30, freq_mask_param=15):
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def add_background_noise(self, waveform):
        noise = torch.randn_like(torch.from_numpy(waveform)) * self.noise_level
        return torch.add(torch.from_numpy(waveform), noise)
    

    def pitch_shift(self, waveform):
        return librosa.effects.pitch_shift(y=waveform, sr=self.sample_rate, n_steps=random.randint(-6, 6))
    
    # sovrapposizione tra due file di input
    def superimpose(self, waveform, random_waveform):
        return torch.add(torch.from_numpy(waveform), torch.from_numpy(random_waveform*0.5))

    
    def augment(self, waveform, random_waveform):
        augmentations = [
            lambda x,_: self.add_background_noise(x),
            lambda x,_: self.pitch_shift(x),
            lambda x,y: self.superimpose(x,y)
        ]
        random.shuffle(augmentations)
        for augment in augmentations[:1]:  
            waveform = augment(waveform, random_waveform)
        return waveform


# Constructing dataset
class EmotionDataset(Dataset):
    def __init__(self, df, processor, augmenter, attention_mask):
        self.df = df
        self.processor = processor
        self.augmenter = augmenter
        self.sample_rate = 16000
        self.max_seconds = 6  #max padding seconds
        self.threshold = 0.8  #max percentage of which files to keep
        self.attention_mask = attention_mask


    def __len__(self):
        return len(self.df)


    # Normalize waveform between 0 and 1
    def normalize_waveform(self, wav_data):
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.float()  
        elif isinstance(wav_data, np.ndarray):
            wav_data = wav_data.astype(np.float32)  
            wav_data = torch.from_numpy(wav_data)  
        
        max_val = wav_data.abs().max()
        if max_val > 0:
            wav_data = wav_data / max_val
        
        return wav_data.numpy() if isinstance(wav_data, torch.Tensor) else wav_data
    
    # Method for retrieving mel coefficients
    @staticmethod
    def get_mel_spectrogram(input_values):
        n_mels = 48
        hop_length = int(0.010 * 16000)  
        win_length = int(0.025 * 16000)
        n_fft = 512  
        mel_spectrogram = librosa.feature.melspectrogram(y=input_values.numpy(), sr=16000, n_mels=n_mels, \
                                                         hop_length=hop_length, win_length=win_length, fmax=8000,\
                                                         n_fft=n_fft)
        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_derivative_1 = librosa.feature.delta(mel_spectrogram, order=1)
        mel_spectrogram_derivative_2 = librosa.feature.delta(mel_spectrogram, order=2)
    
        mel_spectrogram = librosa.util.normalize(mel_spectrogram)
        mel_spectrogram_derivative_1 = librosa.util.normalize(mel_spectrogram_derivative_1)
        mel_spectrogram_derivative_2 = librosa.util.normalize(mel_spectrogram_derivative_2)

        mel_spectrogram_stack = np.stack([mel_spectrogram, mel_spectrogram_derivative_1, mel_spectrogram_derivative_2], axis=0)

        return torch.tensor(mel_spectrogram_stack, dtype=torch.float32)


    def retrieve_random_waveform(self, wav_data):
        random_wav = self.df.iloc[random.randint(0, len(self.df)-1)]["wav_file"]

        return (torch.randn_like(torch.from_numpy(wav_data)) * 0.01).numpy() if len(random_wav) < len(wav_data) \
            else random_wav[:len(wav_data)]


    # Padding of max_seconds and creation of the batch
    def __getitem__(self, idx):
        wav_data = self.df.iloc[idx]["wav_file"]  
        valence = self.df.iloc[idx]["Valence"]
        arousal = self.df.iloc[idx]["Arousal"]

        max_length = self.sample_rate * self.max_seconds

        if len(wav_data) > max_length/ self.threshold:
            return self.__getitem__((idx + 1) % len(self.df))
        
        rand_augmenter = int(random.random()*1000)
        random_wav = self.retrieve_random_waveform(wav_data)

        # Apply file augmentation randomly and occasionaly 
        if self.augmenter and (rand_augmenter%4==0):
            wav_data = self.augmenter.augment(wav_data, random_wav)

        inputs = self.processor(wav_data, sampling_rate=self.sample_rate, return_tensors="pt", padding = 'max_length', \
                                truncation = True, max_length = max_length, do_normalize = True,\
                                return_attention_mask = self.attention_mask)
        
        input_values = inputs['input_values'].squeeze(0)

        inputs['input_values'] = input_values
        inputs['mel_spectrogram'] = EmotionDataset.get_mel_spectrogram(input_values)
        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)

        return inputs

    
class EmotionModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(self.config)
        
        # Freezing the CNN extractors
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
        
        # Fine-tuning of transformer layers
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True
        
        # Mel CNN
        self.mel_cnn = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Flatten()
        )

    
        # BLSTM
        #config.hidden_size = 768
        self.rnn = nn.LSTM(input_size= 4368, hidden_size=config.hidden_size, num_layers=2, \
                           batch_first=True, bidirectional=True, dropout=0.5)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        # Final Regressor
        self.regressor = nn.Linear(self.rnn.hidden_size*2, config.num_labels)
        
        self.init_weights()


    def forward(
            self,
            input_values,
            mel_spectrogram
        ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.mean(hidden_states, dim=1)
        mel_features = self.mel_cnn(mel_spectrogram)
    
        # Combine features
        combined_features = torch.cat((hidden_states, mel_features), dim=1)
        #combined_features = self.dropout(combined_features)
        temp,_ = self.rnn(combined_features)
        temp = self.dropout(temp)
        temp = self.act(temp)
        
        logits = self.regressor(temp)
        
        return hidden_states, logits
    


# Saving best epoch model 
def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch + 1}")


# Dynamically set device (CUDA GPU or CPU)
def return_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# CCC loss
def ccc_loss(gold, pred):
    ccc = ConcordanceCorrCoef().to("cuda")
    coeff = ccc(gold, pred)
    # print("CCC:", coeff)
    ccc_loss = 1 - coeff
    return ccc_loss


def L1(gold, pred):
    #return torch.mean(torch.abs(gold-pred))
    loss = nn.L1Loss()
    return loss(pred, gold)

def L2(gold, pred):
    #return torch.mean((gold-pred)**2)
    loss = nn.MSELoss()
    return loss(pred, gold)

def R2(gold, pred):
    num = torch.sum((gold-pred)**2)
    den = torch.sum((gold - torch.mean(gold))**2)
    return 1 - (num / den)


def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    mel_spectrogram = batch['mel_spectrogram'].to(device)
    labels = batch['labels'].to(device)

    return input_values, labels, mel_spectrogram



def compute_loss(model, device, batch, alpha, beta):
    input_values, labels,  mel_spectrogram = batch_values(batch, device)  

    # For small batch sizes where variance could be very low
    if labels[:, 0].std() < 1e-7 or labels[:, 1].std() < 1e-7:
        print("Value equal to 0 or invariance in labels!")
        return None
    
    _,logits = model(input_values, mel_spectrogram)

    loss_val = ccc_loss(labels[:, 0], logits[:, 0])
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1])

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")
    return loss, loss_val, loss_ar, labels, logits



# Training function
def train(model, device, train_dataloader, test_dataloader, \
          epochs=3, alpha=0.5, beta=0.5, checkpoint_path = "custom_model.pth", patience_es = 15):
    
    print("****TRAINING****")
    train_losses = []
    val_losses = []
    valence_losses = []
    arousal_losses = []
    l1_losses_val = []
    l2_losses_val = []
    r2_losses_val = []
    l1_losses_ar = []
    l2_losses_ar = []
    r2_losses_ar = []
    best_val_loss = float("inf")
    no_improvement_epochs = 0
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_dataloader), epochs=10)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training Loop
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()
           
            loss, _, _, _, _, = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue 
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            

            optimizer.step()
            loss = loss.item()
            epoch_loss += loss
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss}")

        # Validation Loop
        val_loss, loss_val, loss_ar, l1_val, l2_val, r2_val, l1_ar, l2_ar, r2_ar = validate(model, device, test_dataloader, alpha, beta)

        val_losses.append(val_loss)
        valence_losses.append(loss_val)
        arousal_losses.append(loss_ar)
        l1_losses_val.append(l1_val)
        l2_losses_val.append(l2_val)
        r2_losses_val.append(r2_val)
        l1_losses_ar.append(l1_ar)
        l2_losses_ar.append(l2_ar)
        r2_losses_ar.append(r2_ar)

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"\tNew best model saved with Validation Loss: {val_loss:.4f}")
        else:
            no_improvement_epochs += 1
            print(f"\tNo improvement for {no_improvement_epochs} epochs...")

        # Early Stopping Check
        if no_improvement_epochs >= patience_es:
            print("-----------EARLY STOPPING TRIGGERED.-----------")
            break
        
        plot_losses(train_losses, val_losses)
        plot_metrics(valence_losses, arousal_losses, l1_losses_val, l2_losses_val, r2_losses_val, \
                     l1_losses_ar, l2_losses_ar, r2_losses_ar)
        scheduler.step(val_loss)


# Validation Loop
def validate(model, device, test_dataloader, alpha, beta):
    model.eval()
    
    val_loss = 0
    val_loss_val = 0
    val_loss_ar = 0

    labels = None
    logits = None
    print("****VALIDATION****")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss, loss_val, loss_ar, lab, log = compute_loss(model, device, batch, alpha, beta)
            if lab is not None:
                if labels is None:
                    labels = lab.to(return_device())
                    logits = log.to(return_device())  
                else:
                    labels = torch.cat((labels, lab))
                    logits = torch.cat((logits, log)) 

            if loss is None: continue

            val_loss += loss.item()
            val_loss_val += loss_val.item()
            val_loss_ar += loss_ar.item()


    # Average CCC scores
    avg_val_loss = val_loss / len(test_dataloader)
    avg_val_loss_val = val_loss_val / len(test_dataloader)
    avg_val_loss_ar = val_loss_ar / len(test_dataloader)


    l1_val = L1(labels[:,0], logits[:,0]).item()
    l2_val = L2(labels[:,0], logits[:,0]).item()
    r2_val = R2(labels[:,0], logits[:,0]).item()

    l1_ar = L1(labels[:,1], logits[:,1]).item() 
    l2_ar = L2(labels[:,1], logits[:,1]).item()
    r2_ar = R2(labels[:,1], logits[:,1]).item()
    
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss, avg_val_loss_val, avg_val_loss_ar, l1_val, l2_val, r2_val, l1_ar, l2_ar, r2_ar


def plot_losses(train_losses, val_losses, filename = "plots/loss_plot_trial.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    

def plot_metrics(valence_losses, arousal_losses, l1_val, l2_val, r2_val, l1_ar, l2_ar, r2_ar, filename = "plots/metrics_plot_trial.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(valence_losses) + 1), valence_losses, label='Valence CCC Loss', marker='o')
    plt.plot(range(1, len(arousal_losses) + 1), arousal_losses, label='Arousal CCC Loss', marker='o')
    plt.plot(range(1, len(l1_val) + 1), l1_val, label='L1 Val Loss', marker='o')
    plt.plot(range(1, len(l2_val) + 1), l2_val, label='L2 Val Loss', marker='o')
    plt.plot(range(1, len(r2_val) + 1), r2_val, label='R2 Val Loss', marker='o')
    plt.plot(range(1, len(l1_ar) + 1), l1_ar, label='L1 Ar Loss', marker='o')
    plt.plot(range(1, len(l2_ar) + 1), l2_ar, label='L2 Ar Loss', marker='o')
    plt.plot(range(1, len(r2_ar) + 1), r2_ar, label='R2 Ar Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

# Load Checkpoint
def load_trained_model(device, checkpoint_path, pretrained_model):
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    model = EmotionModel(config).to(device)
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    print(checkpoint_path)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model from checkpoint.")
    else:
        print("Checkpoint not found. Using untrained model.")
    
    return model, processor


# Prediction
def predict_emotion(model, device, processor, wav_data):
    model.eval()
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding = 'max_length', \
                                truncation = True, max_length = 6*16000, do_normalize = True,\
                                return_attention_mask = False)

    input_values = inputs['input_values'].to(device)
    mel_spectrogram = EmotionDataset.get_mel_spectrogram(input_values).to(device)
    mel_spectrogram = mel_spectrogram.permute(1,0,2,3)


    with torch.no_grad():
        _, outputs = model(input_values=input_values, mel_spectrogram=mel_spectrogram)

    return outputs




def main():
    device = return_device()
    
    pretrained_model = "facebook/wav2vec2-base" 
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model, attn_implementation="flash_attention_2")
    config = Wav2Vec2Config.from_pretrained(pretrained_model)

    muse = pd.read_pickle("data/MuSe_sample").sample(frac=1, random_state=42)
    iemocap = pd.read_pickle("data/IEMOCAP_useful").sample(frac=1, random_state=42)
    msp = pd.read_pickle("data/MSP_PODCAST_SAMPLED").sample(frac=1, random_state=42)
    df = pd.concat([iemocap, muse, msp]).sample(frac=1, random_state=42)
    
    print(df["Valence"].describe())
    print(df["Arousal"].describe())

    
    df.drop(columns = ["Name"], inplace = True)
    print(df)


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    augmenter = AudioAugmentation(sample_rate=16000)

    att_mask = False
    if config.feat_extract_norm == "layer":
        print("\tReturn Attention Mask")
        att_mask = True
    
    train_dataset = EmotionDataset(train_df, processor, augmenter, att_mask)
    test_dataset = EmotionDataset(test_df, processor, augmenter, att_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True, )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True,)

    del df, train_df, test_df, train_dataset, test_dataset, muse, iemocap, msp
    gc.collect()
    
    model = EmotionModel(config).to(device)
    summary(model)
    train(model, device, train_dataloader, test_dataloader, epochs = 50)


if __name__ == "__main__":
    main()



