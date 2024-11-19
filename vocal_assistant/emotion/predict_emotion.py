import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Config
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torchmetrics.regression import ConcordanceCorrCoef
import gc

class EmotionDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_data = self.df.iloc[idx]["wav_file"]  
        valence = self.df.iloc[idx]["Valence"]
        arousal = self.df.iloc[idx]["Arousal"]
        #dominance = self.df.iloc[idx]["Dominance"]
        
        # Process audio input
        inputs = self.processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)
        
        return inputs
    

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)        
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
            attention_mask
    ):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits
    


"""class Wav2Vec2ForEmotionRegression(nn.Module):
    def __init__(self):
        super(Wav2Vec2ForEmotionRegression, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.rnn = nn.LSTM(input_size=self.wav2vec2.config.hidden_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout()
        self.regression_layer = nn.Linear(128 * 2, 3)  # 128 * 2 because it's bidirectional
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        drop_out = self.dropout(hidden_states)
        rnn_output, _ = self.rnn(drop_out)

        return self.regression_layer(rnn_output)"""

def custom_collate(batch):
    input_values = [item['input_values'].squeeze(0) for item in batch]
    attention_mask = [item['attention_mask'].squeeze(0) if 'attention_mask' in item else torch.ones_like(item['input_values'].squeeze(0)) for item in batch]
    labels = torch.stack([item['labels'] for item in batch])

    # Pad input values and attention mask to the longest in the batch
    input_values_padded = pad_sequence(input_values, batch_first=True)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True)

    return {
        'input_values': input_values_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels
    }

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch + 1}")

"""def load_checkpoint(model, optimizer, filename):
    device = return_device()
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)  
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0"""
    
def return_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dynamically set device

def ccc(gold, pred):
    gold = gold.squeeze(-1)
    pred = pred.squeeze(-1)
    
    gold_mean = torch.mean(gold, dim=-1, keepdim=True)
    pred_mean = torch.mean(pred, dim=-1, keepdim=True)
    
    covariance = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=-1, keepdim=True)
    gold_var = torch.mean((gold - gold_mean) ** 2, dim=-1, keepdim=True)
    pred_var = torch.mean((pred - pred_mean) ** 2, dim=-1, keepdim=True)

    pearson_coefficient = covariance/(torch.sqrt(gold_var) * torch.sqrt(pred_var))
    
    ccc = (2 * pearson_coefficient * covariance) / (gold_var + pred_var + (gold_mean - pred_mean) ** 2 + 1e-7)
    return ccc

def ccc_loss(gold, pred):
    ccc_loss = 1 - ccc(gold, pred)
    return ccc_loss

def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    return input_values, attention_mask, labels

def train(model, train_dataloader, test_dataloader, epochs=3, alpha=0.5, beta=0.5):
    """
    Train the model using CCC loss for valence and arousal.
    """
    device = return_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    checkpoint_path = "model_checkpoint_sampled.pth"

    for epoch in range(epochs):
        model.gradient_checkpointing_enable()
        model.train()
        epoch_loss = 0
        gc.collect()

        # Training Loop
        for batch in tqdm(train_dataloader):
            input_values, attention_mask, labels = batch_values(batch, device)
            
            optimizer.zero_grad()
            _, logits = model(input_values=input_values, attention_mask=attention_mask)
            print("logits:", logits)
            
            # Compute CCC Loss for valence and arousal
            loss_val = ccc_loss(labels[:, 0], logits[:, 0]).mean()
            loss_ar = ccc_loss(labels[:, 1], logits[:, 1]).mean()

            # Weighted total loss
            loss = alpha * loss_val + beta * loss_ar
            print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(train_dataloader)}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Validation Loop
        validate(model, test_dataloader, alpha, beta)


# Validation Loop
def validate(model, test_dataloader, alpha, beta):
    device = return_device()
    model.eval()
    val_loss = 0
    ccc_scores = {"valence": [], "arousal": []}

    with torch.no_grad():
        for batch in test_dataloader:
            input_values, attention_mask, labels = batch_values(batch, device)
            
            _, logits = model(input_values=input_values, attention_mask=attention_mask)
            print("logits:", logits)
            
            # Compute CCC Loss
            loss_val = ccc_loss(labels[:, 0], logits[:, 0]).mean()
            loss_ar = ccc_loss(labels[:, 1], logits[:, 1]).mean()
            
            # Store raw CCC scores
            ccc_scores["valence"].append((1 - loss_val).item())  # Raw CCC
            ccc_scores["arousal"].append((1 - loss_ar).item())  # Raw CCC
            
            # Total weighted loss
            loss = alpha * loss_val + beta * loss_ar
            val_loss += loss.item()

    # Average CCC scores
    avg_ccc_val = sum(ccc_scores["valence"]) / len(ccc_scores["valence"])
    avg_ccc_ar = sum(ccc_scores["arousal"]) / len(ccc_scores["arousal"])

    print(f"Validation Loss: {val_loss / len(test_dataloader)}")
    print(f"Average CCC - Valence: {avg_ccc_val:.4f}, Arousal: {avg_ccc_ar:.4f}")


def load_trained_model(checkpoint_path, pretrained_model):
    device = return_device()
    #model = Wav2Vec2ForEmotionRegression().to(device)
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    config.num_labels = 2  # Ensure this matches the number of regression outputs (Valence, Arousal, Dominance)
    model = EmotionModel(config).to(device)
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model from checkpoint.")
    else:
        print("Checkpoint not found. Using untrained model.")
    
    return model, processor

def predict_emotion(model, processor, wav_data):
    device = return_device()
    model.eval()
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].to(device)
    attention_mask = inputs.get('attention_mask').to(device) if 'attention_mask' in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
    
    return outputs[1]

#if __name__ == "__main ":
pretrained_model = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
#model = Wav2Vec2ForEmotionRegression().to(return_device())
config = Wav2Vec2Config.from_pretrained(pretrained_model)
config.num_labels = 2  # Ensure this matches the number of regression outputs (Valence, Arousal, Dominance)
#cuda2 = torch.device('cuda:1')
model = EmotionModel(config).to('cpu')

df = pd.read_pickle("data/MuSe_useful")
df_sampled = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
print(df_sampled.head())

train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=42)

# Create datasets
train_dataset = EmotionDataset(train_df, processor)
test_dataset = EmotionDataset(test_df, processor)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=4)

torch.cuda.empty_cache()  # Releases unoccupied cached memory.
torch.cuda.reset_peak_memory_stats()  # Resets memory stats for accurate debugging.
train(model, train_dataloader, test_dataloader, epochs = 5)

#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf