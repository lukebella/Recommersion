import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import os

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
        dominance = self.df.iloc[idx]["Dominance"]
        
        # Process audio input
        inputs = self.processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs['labels'] = torch.tensor([valence, arousal, dominance], dtype=torch.float32)
        
        return inputs

class Wav2Vec2ForEmotionRegression(nn.Module):
    def __init__(self):
        super(Wav2Vec2ForEmotionRegression, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.rnn = nn.LSTM(input_size=self.wav2vec2.config.hidden_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.regression_layer = nn.Linear(128 * 2, 3)  # 128 * 2 because it's bidirectional

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        rnn_output, _ = self.rnn(hidden_states)
        
        # Average pooling over the time dimension
        pooled_output = rnn_output.mean(dim=1)
        return self.regression_layer(pooled_output)

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

def load_checkpoint(model, optimizer, filename):
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
        return 0
    
def return_device():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dynamically set device

def train(model, train_dataloader, test_dataloader, epochs=3):
    device = return_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss() 
    checkpoint_path = "model_checkpoint_sampled.pth"

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in tqdm(train_dataloader):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
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
                
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(test_dataloader)}")


def load_trained_model(checkpoint_path):
    device = return_device()
    model = Wav2Vec2ForEmotionRegression().to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model from checkpoint.")
    else:
        print("Checkpoint not found. Using untrained model.")
    
    return model, processor

def predict_emotion(model, processor, wav_data):
    model.eval()
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].to("cpu")
    attention_mask = inputs.get('attention_mask').to("cpu") if 'attention_mask' in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
    
    valence, arousal, dominance = outputs.squeeze().tolist()
    return {
        "Valence": valence,
        "Arousal": arousal,
        "Dominance": dominance
    }

if __name__ == "__main ":
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForEmotionRegression().to(return_device())

    df = pd.read_pickle("data/IEMOCAP_useful")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_sampled = df_shuffled.sample(frac=0.01, random_state=42)

    train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = EmotionDataset(train_df, processor)
    test_dataset = EmotionDataset(test_df, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    train(model, train_dataloader, test_dataloader, epochs = 3)
