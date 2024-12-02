import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Config
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
import matplotlib.pyplot as plt


class EmotionDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.sample_rate = 16000

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_data = self.df.iloc[idx]["wav_file"]  
        valence = self.df.iloc[idx]["Valence"]
        arousal = self.df.iloc[idx]["Arousal"]
        
        wav_data = wav_data[:self.sample_rate*10]
        # Process audio input
        inputs = self.processor(wav_data, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)
        return inputs
    

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.rnn = nn.LSTM(input_size= config.hidden_size, hidden_size=config.hidden_size, num_layers=2, \
                           batch_first=True, bidirectional=True)        
        self.dropout = nn.Dropout(config.final_dropout)
        self.dense = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        #x = self.dropout(x)
        x,_ = self.rnn(x)
        x = torch.tanh(x)        
        #x = self.dropout(x)
        x = self.dense(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(self.config)
        self.init_weights()
        self.gradient_checkpointing_enable()
        self.config = config
        """for param in self.wav2vec2.parameters():
            param.requires_grad = False"""
        self.classifier = RegressionHead(config)
        


    def forward(
            self,
            input_values,
            attention_mask
    ):
        #print("INPUT_VALUES:",sys.getsizeof(input_values))
        #print("SUMMARY:",torch.cuda.memory_summary())
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        #print("OUTPUTS_VALUES:",sys.getsizeof(outputs))
        #print("SUMMARY:",torch.cuda.memory_summary())
        hidden_states = outputs[0]
        #print("hidden states:",sys.getsizeof(hidden_states))
        #print("SUMMARY:",torch.cuda.memory_summary())
        hidden_states = torch.mean(hidden_states, dim=1)
        #print("hidden states:",sys.getsizeof(hidden_states))
        #print("SUMMARY:",torch.cuda.memory_summary())
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def custom_collate(batch):
    input_values = [item['input_values'].squeeze(0) for item in batch]
    attention_mask = [item['attention_mask'].squeeze(0) if 'attention_mask' \
                      in item else torch.ones_like(item['input_values'].squeeze(0)) for item in batch]
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

    #pearson_coefficient = covariance/(torch.sqrt(gold_var) * torch.sqrt(pred_var))
    
    ccc = (2 * covariance) / (gold_var + pred_var + (gold_mean - pred_mean) ** 2 + 1e-7)
    return ccc

def ccc_loss(gold, pred):
    ccc_loss = 1 - ccc(gold, pred)
    return ccc_loss


def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    return input_values, attention_mask, labels



def compute_loss(model, device, batch, alpha, beta):
    input_values, attention_mask, labels = batch_values(batch, device)
   
    #TODO Resolve this issue
    if (labels[:, 0].var() == 0 or labels[:, 1].var() == 0):
        print("Value equal to 0 or invariance in labels!")
        return None

    _, logits = model(input_values=input_values, attention_mask=attention_mask)
    
    
    # Compute CCC Loss for valence and arousal
    loss_val = ccc_loss(labels[:, 0], logits[:, 0])
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1])

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")

    return loss



def train(model, device, train_dataloader, test_dataloader, optimizer, epochs=3, alpha=0.5, beta=0.5, checkpoint_path = "model_checkpoint_sampled.pth",):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    print("****TRAINING****")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training Loop
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()
           
            loss = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue 

            # Backpropagation
            loss.backward()
        
            optimizer.step()
            loss = loss.item()
            # avg loss over all processes
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Validation Loop
        val_loss = validate(model, device, test_dataloader, alpha, beta)
        val_losses.append(val_loss)

        #TODO undersand why in the last step there is a mismatch between batch and input
    plot_losses(train_losses, val_losses)


# Validation Loop
def validate(model, device, test_dataloader, alpha, beta):
    model.eval()
    avg_val_loss = 0
    val_loss = 0
    print("****VALIDATION****")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue
            print(loss)
            val_loss += loss.item()

    # Average CCC scores
    avg_val_loss = val_loss / len(test_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss


def plot_losses(train_losses, val_losses, filename = "./loss_plot_1.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(filename)  # Save the plot to a file
    print(f"Plot saved as {filename}")


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
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].to(device)
    attention_mask = inputs.get('attention_mask').to(device) if 'attention_mask' in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
    
    return outputs[1]


def main():
    device = return_device()
    
    pretrained_model = "facebook/wav2vec2-base"    #patrickvonplaten/wav2vec2_tiny_random_robust" #
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    model = EmotionModel(config).to(device)

    df = pd.read_pickle("data/full_data").sample(frac=1, random_state=42).reset_index(drop=True)
    print(df.head())

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    train_dataset = EmotionDataset(train_df, processor)
    test_dataset = EmotionDataset(test_df, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,\
                                collate_fn=custom_collate, \
                                num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle=True,\
                                collate_fn=custom_collate, \
                                num_workers=4, pin_memory=True)

    summary(model)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    train(model, device, train_dataloader, test_dataloader, optimizer, epochs = 20)


if __name__ == "__main__":
    main()



#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf
#when the user will test the model, try to:
# - mix the two dataset for training (full_data)
# - train with iemocap and test with muse
# - train with muse and test with iemocap
# - Make in the interface a selector for these three different models and check which is the most useful 


#For execute it: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 vocal_assistant/emotion/predict_emotion.py
