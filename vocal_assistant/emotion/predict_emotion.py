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
        #self.df["Valence"] = (self.df["Valence"] - self.df["Valence"].min()) / (self.df["Valence"].max() - self.df["Valence"].min())
        #self.df["Arousal"] = (self.df["Arousal"] - self.df["Arousal"].min()) / (self.df["Arousal"].max() - self.df["Arousal"].min())

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

        self.dense = nn.Linear(config.hidden_size, config.num_labels)
        #self.dropout = nn.Dropout(config.final_dropout)
        #self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        #x = self.dropout(x)
        x = self.dense(x)
        #x = torch.tanh(x)
        #x = self.dropout(x)
        #x = self.out_proj(x)

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
        hidden_states = hidden_states.mean(dim=1)
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

def train(model, train_dataloader, test_dataloader, epochs=3):
    device = return_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    #loss_fn = nn.MSELoss()
    #loss_fn = nn.SmoothL1Loss() 
    loss_fn = ConcordanceCorrCoef(num_outputs=1).to(device)
    checkpoint_path = "model_checkpoint_sampled.pth"

    model.train()

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        epoch_loss = 0
        gc.collect()
        for batch in tqdm(train_dataloader):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            #outputs = model(input_values=input_values, attention_mask=attention_mask)
            #loss = loss_fn(outputs, labels)
            _, logits = model(input_values=input_values, attention_mask=attention_mask)
            print("logits_val: ", logits[:, 0])
            print("labels_val: ", labels[:, 0])
            print("logits_ar: ", logits[:, 1])
            print("labels_ar: ", labels[:, 1])
            if (labels[:, 0].var()==0 or labels[:, 1].var() == 0):
                print("Labels var = 0!")
                continue
            loss_val = loss_fn(logits[:, 0], labels[:, 0])
            print("Loss Val: ",loss_val)
            loss_ar = loss_fn(logits[:, 1], labels[:, 1])
            print("Loss Ar: ",loss_ar)
            #loss_dom = loss_fn(logits[:, 2], labels[:, 2])
            #print(loss_val, loss_ar, loss_dom)
            loss = torch.mean(torch.stack([loss_val, loss_ar])) # stack and mean for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            print(loss)
            epoch_loss += loss.item()
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

        
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(train_dataloader)}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_values = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                #outputs = model(input_values=input_values, attention_mask=attention_mask)
                #loss = loss_fn(outputs, labels)
                _, logits = model(input_values=input_values, attention_mask=attention_mask)
                loss_val = loss_fn(logits[:, 0], labels[:, 0])
                loss_ar = loss_fn(logits[:, 1], labels[:, 1])
                if (labels[:, 0].var()==0 or labels[:, 1].var() == 0):
                    print("Labels var = 0!")
                    continue
                #loss_dom = loss_fn(logits[:, 2], labels[:, 2])
                #print(loss_val, loss_ar, loss_dom)
                loss = torch.mean(torch.stack([loss_val, loss_ar]))  # stack and mean for stability
                val_loss += loss.item()
                torch.cuda.empty_cache()


        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(test_dataloader)}")


def load_trained_model(checkpoint_path, pretrained_model):
    device = return_device()
    #model = Wav2Vec2ForEmotionRegression().to(device)
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    config.num_labels = 3  # Ensure this matches the number of regression outputs (Valence, Arousal, Dominance)
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
    
    """valence, arousal, dominance = outputs.squeeze().tolist()
    return {
        "Valence": valence,
        "Arousal": arousal,
        "Dominance": dominance
    }"""
    return outputs[1]

#if __name__ == "__main ":
pretrained_model = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
#model = Wav2Vec2ForEmotionRegression().to(return_device())
config = Wav2Vec2Config.from_pretrained(pretrained_model)
config.num_labels = 2  # Ensure this matches the number of regression outputs (Valence, Arousal, Dominance)
model = EmotionModel(config).to(return_device())
model.gradient_checkpointing_enable()


df = pd.read_pickle("data/full_data")
print(df.shape)
#assert not df[["Valence", "Arousal"]].isnull().values.any(), "NaNs in target labels"
df_sampled = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df_sampled.head())

train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=42)

# Create datasets
train_dataset = EmotionDataset(train_df, processor)
test_dataset = EmotionDataset(test_df, processor)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=4)

"""for batch in train_dataloader:
    first_input_values = batch['input_values']  # Access first element of 'input_values' in the batch
    print(first_input_values.shape)"""
#print("Dataset Variance: "+torch.var(train_dataloader, unbiased=True))
torch.cuda.empty_cache()  # Releases unoccupied cached memory.
torch.cuda.reset_peak_memory_stats()  # Resets memory stats for accurate debugging.
train(model, train_dataloader, test_dataloader, epochs = 5)
