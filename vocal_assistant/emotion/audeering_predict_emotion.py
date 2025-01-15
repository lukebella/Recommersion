import os
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
from torchmetrics.regression import ConcordanceCorrCoef
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchaudio
import random
import librosa
from sklearn.model_selection import KFold


class AudioAugmentation:
    def __init__(self, sample_rate=16000, noise_level=0.005, time_mask_param=30, freq_mask_param=15):
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def add_background_noise(self, waveform):
        noise = torch.randn_like(torch.from_numpy(waveform)) * self.noise_level
        return torch.add(torch.from_numpy(waveform), noise)

    def time_stretch(self, waveform, rate=1.1):
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        stretched = torchaudio.transforms.TimeStretch()(spectrogram)
        return torch.tensor(stretched)

    def pitch_shift(self, waveform):
        return librosa.effects.pitch_shift(y=waveform, sr=self.sample_rate, n_steps=random.randint(-6, 6))

    
    def augment(self, waveform):
        augmentations = [
            #self.add_background_noise,
            #lambda x: self.time_stretch(x, rate=random.uniform(0.8, 1.2)),
            lambda x: self.pitch_shift(x),
        ]
        random.shuffle(augmentations)
        for augment in augmentations[:1]:  # Apply 2 random augmentations
            waveform = augment(waveform)
        return waveform



class EmotionDataset(Dataset):
    def __init__(self, df, processor, augmenter, attention_mask):
        self.df = df
        self.processor = processor
        self.augmenter = augmenter
        self.sample_rate = 16000
        self.max_seconds = 5  #max padding seconds
        self.threshold = 0.8  #max percentage of which files to keep
        self.attention_mask = attention_mask


    def __len__(self):
        return len(self.df)


    def only_vocals(self, waveform):
        S_full, phase = librosa.magphase(librosa.stft(waveform))
        S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',)
                                        #width=int(librosa.time_to_frames(2, sr=self.sample_rate)))

        S_filter = np.minimum(S_full, S_filter)
        margin_v = 2
        power = 2

        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)
        
        S_foreground = mask_v * S_full
        return librosa.istft(S_foreground * phase)



    def normalize_waveform(self, wav_data):
        """
        Normalize audio waveform to the range [-1, 1].
        Handles both torch.Tensor and numpy.ndarray.
        """
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.float()  # Ensure float type for PyTorch tensors
        elif isinstance(wav_data, np.ndarray):
            wav_data = wav_data.astype(np.float32)  # Ensure float32 type for NumPy arrays
            wav_data = torch.from_numpy(wav_data)  # Convert to torch.Tensor for consistency
        
        max_val = wav_data.abs().max()
        if max_val > 0:
            wav_data = wav_data / max_val
        
        return wav_data.numpy() if isinstance(wav_data, torch.Tensor) else wav_data
    

    @staticmethod
    def get_mel_spectrogram(input_values):
        #hop_length = int(0.012 * self.sample_rate)  
        #win_length = int(0.026 * self.sample_rate)  
        mel_spectrogram = librosa.feature.melspectrogram(y=input_values.numpy(), sr=16000, n_mels=40, fmax=8000)
        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_derivative_1 = librosa.feature.delta(mel_spectrogram, order=1)
        mel_spectrogram_derivative_2 = librosa.feature.delta(mel_spectrogram, order=2)

    
        mel_spectrogram = librosa.util.normalize(mel_spectrogram)
        mel_spectrogram_derivative_1 = librosa.util.normalize(mel_spectrogram_derivative_1)
        mel_spectrogram_derivative_2 = librosa.util.normalize(mel_spectrogram_derivative_2)

        mel_spectrogram_stack = np.stack([mel_spectrogram, mel_spectrogram_derivative_1, mel_spectrogram_derivative_2], axis=0)
        #print(mel_spectrogram_stack.shape)

        return torch.tensor(mel_spectrogram_stack, dtype=torch.float32)


    
    def __getitem__(self, idx):
        wav_data = self.df.iloc[idx]["wav_file"]  
        valence = self.df.iloc[idx]["Valence"]
        arousal = self.df.iloc[idx]["Arousal"]
        
        max_length = self.sample_rate * self.max_seconds

        if len(wav_data) > max_length/ self.threshold:
            return self.__getitem__((idx + 1) % len(self.df))
        
        #wav_data = self.normalize_waveform(wav_data)
        #wav_data = self.only_vocals(wav_data)
        """rand_augmenter = int(random.random()*1000)

        if self.augmenter and (rand_augmenter%3==0):
            wav_data = self.augmenter.augment(wav_data)"""

        inputs = self.processor(wav_data, sampling_rate=self.sample_rate, return_tensors="pt", padding = 'max_length', \
                                truncation = True, max_length = max_length, do_normalize = True,\
                                return_attention_mask = self.attention_mask)
        
        #print(inputs['input_values'])
        input_values = inputs['input_values'].squeeze(0)

        inputs['input_values'] = input_values
        inputs['mel_spectrogram'] = EmotionDataset.get_mel_spectrogram(input_values)
        #print("MEL SIZE",inputs['mel_spectrogram'].size())

        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)


        return inputs

    
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(self.config)
        
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
        
        # Allow fine-tuning of transformer layers
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True

        """self.rnn = nn.LSTM(input_size= 4528, hidden_size=2,\
                           batch_first=True, bidirectional=False)"""
        
        
        self.mel_cnn = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten()
        )

        self.rnn = nn.LSTM(input_size= 4144, hidden_size=config.hidden_size, num_layers=2, \
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        self.dropout = nn.Dropout(0.5)
        self.regressor = nn.Linear(config.hidden_size*2, config.num_labels)
        #self.act = nn.ReLU()

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
        # print(mel_features.size())
        # print(hidden_states.size())
        combined_features = torch.cat((hidden_states, mel_features), dim=1)
        #print(combined_features.size())
        #combined_features = self.dropout(combined_features)
        temp,_ = self.rnn(combined_features)
        #temp = self.dropout(temp)
        #temp = self.act(temp)
        logits = self.regressor(temp)
        
        return hidden_states, logits


writer = SummaryWriter("runs/emotion_model")

def log_embedding_norms(model, epoch):
    """ Log the L2 norm degli embedding del modello."""
    for name, param in model.named_parameters():
        if "wav2vec2.encoder" in name and param.requires_grad:
            embedding_norm = param.norm(2).item()
            writer.add_scalar(f"Embedding Norm/{name}", embedding_norm, epoch)


def log_gradient_norms(model, epoch):
    """ Log la norma dei gradienti durante l'addestramento."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            writer.add_scalar(f"Gradient Norm/{name}", grad_norm, epoch)


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch + 1}")

    
def return_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dynamically set device


def ccc_loss(gold, pred):
    ccc = ConcordanceCorrCoef().to("cuda")
    coeff = ccc(gold, pred)
    print("CCC:", coeff)
    ccc_loss = 1 - coeff
    return ccc_loss


def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    mel_spectrogram = batch['mel_spectrogram'].to(device)
    labels = batch['labels'].to(device)

    return input_values, labels, mel_spectrogram



def compute_loss(model, device, batch, alpha, beta):
    input_values, labels,  mel_spectrogram = batch_values(batch, device)  #

    #For small batch sizes where variance could be very low
    if labels[:, 0].std() < 1e-7 or labels[:, 1].std() < 1e-7:
        print("Value equal to 0 or invariance in labels!")
        return None
    
    _,logits = model(input_values, mel_spectrogram)

    #_, logits = model(input_values=input_values)#, attention_mask=attention_mask)
    # Example in validation loop
    print("Predictions:", logits[:8].detach().cpu().numpy())
    print("True labels:", labels[:8].detach().cpu().numpy())

    loss_val = ccc_loss(labels[:, 0], logits[:, 0])
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1])

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")
    return loss, loss_val, loss_ar


def get_gradients(model):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach().cpu().numpy()
    return gradients


def plot_gradients(gradients, layer_name):
    if layer_name in gradients:
        grad = gradients[layer_name]
        plt.hist(grad.flatten(), bins=100)
        plt.title(f'Gradient Distribution - {layer_name}')
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
        plt.savefig("gradients.png")


def train(model, device, train_dataloader, test_dataloader, \
          epochs=3, alpha=0.5, beta=0.5, checkpoint_path = "aud_model.pth", patience_es = 15):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    print("****TRAINING****")
    train_losses = []
    val_losses = []
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
           
            loss, _, _ = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue 

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            log_gradient_norms(model, epoch)
            """gradients = get_gradients(model)

            for name, grad in gradients.items():
                grad_norm = np.linalg.norm(grad)
                print(f"Gradient Norm for {name}: {grad_norm}")"""

            optimizer.step()
            loss = loss.item()
            epoch_loss += loss
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss}")
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        log_embedding_norms(model, epoch)

        # Validation Loop
        val_loss = validate(model, device, test_dataloader, alpha, beta)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        val_losses.append(val_loss)
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
        scheduler.step(val_loss)

    writer.close()
    plot_losses(train_losses, val_losses)

#CV
def cross_validate_alpha_beta(device, dataset, alpha_beta_values, config, k=5, epochs=12):
   
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for alpha, beta in alpha_beta_values:
        print(f"Testing alpha={alpha}, beta={beta}")
        fold_losses = []

        for train_idx, val_idx in kfold.split(dataset):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            # Reinitialize model and optimizer for each fold
            model = EmotionModel(config).to(device)
            optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)

            for epoch in range(epochs):
                # Training loop
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = compute_loss(model, device, batch, alpha, beta)
                    if loss is None: continue
                    loss.backward()
                    optimizer.step()

                # Validation loop
                val_loss = validate(model, device, val_loader, alpha, beta)
                fold_losses.append(val_loss)

        # Store the mean validation loss for the parameter combination
        mean_loss = sum(fold_losses) / len(fold_losses)
        results[(alpha, beta)] = mean_loss
        print(f"Mean Validation Loss for alpha={alpha}, beta={beta}: {mean_loss}")

    # Find the best alpha and beta
    best_params = min(results, key=results.get)
    print(f"Best alpha, beta combination: {best_params} with Loss: {results[best_params]}")
    return best_params


# Validation Loop
def validate(model, device, test_dataloader, alpha, beta):
    model.eval()
    avg_val_loss = 0
    val_loss = 0
    avg_val_loss_val = 0
    val_loss_val = 0
    avg_val_loss_ar = 0
    val_loss_ar = 0
    print("****VALIDATION****")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss, loss_val, loss_ar = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue

            val_loss += loss.item()
            #val_loss_val += loss_val.item()
            #val_loss_ar += loss_ar.item()

    # Average CCC scores
    avg_val_loss = val_loss / len(test_dataloader)
    #avg_val_loss_val = val_loss_val / len(test_dataloader)
    #avg_val_loss_ar = val_loss_ar / len(test_dataloader)

    #TODO print best valence and arousal losses
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss


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
    #plt.show()
    # Save the plot to a file
    


def load_trained_model(device, checkpoint_path, pretrained_model):
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    model = EmotionModel(config).to(device)
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model from checkpoint.")
    else:
        print("Checkpoint not found. Using untrained model.")
    
    return model, processor


def predict_emotion(model, device, processor, wav_data):
    model.eval()
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding = 'max_length', \
                                truncation = True, max_length = 10*16000, do_normalize = True,\
                                return_attention_mask = False)

    input_values = inputs['input_values'].to(device)
    mel_spectrogram = EmotionDataset.get_mel_spectrogram(input_values).to(device)
    mel_spectrogram = mel_spectrogram.permute(1,0,2,3)

    with torch.no_grad():
        _, outputs = model(input_values=input_values, mel_spectrogram=mel_spectrogram)

    
    return outputs#[1]



def main():
    device = return_device()
    
    pretrained_model = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim' #"facebook/wav2vec2-base"    #patrickvonplaten/wav2vec2_tiny_random_robust" #w2v2-L-robust-12
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model, attn_implementation="flash_attention_2")
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    config.num_labels = 2
    muse = pd.read_pickle("data/MuSe_sample").sample(frac=1, random_state=42)#.reset_index(drop=True)
    iemocap = pd.read_pickle("data/IEMOCAP_useful").sample(frac=1, random_state=42)

    df = pd.concat([iemocap, muse]).sample(frac=1, random_state=42)
    
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

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True, )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True,)

    
    #alpha_beta_values = [(a, 1 - a) for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    #best_alpha, best_beta = cross_validate_alpha_beta(device, train_dataset, alpha_beta_values, config)

    model = EmotionModel(config).to(device)
    summary(model)
    train(model, device, train_dataloader, test_dataloader, epochs = 50)#, alpha = best_alpha, beta = best_beta)


if __name__ == "__main__":
    main()


#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf
#when the user will test the model, try to:
# - mix the two dataset for training (full_data)
# - train with iemocap and test with muse
# - train with muse and test with iemocap
# - Make in the interface a selector for these three different models and check which is the most useful 
