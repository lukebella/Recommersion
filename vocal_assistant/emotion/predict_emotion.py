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
import seaborn as sns
from graphviz import Digraph
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class EmotionDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.sample_rate = 16000

    def __len__(self):
        return len(self.df)
    
    def normalize_waveform(self, wav_data):
        """
        Normalize audio waveform to the range [-1, 1].
        """
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.float()  # Ensure float type
        max_val = wav_data.abs().max()
        if max_val > 0:
            wav_data = wav_data / max_val
        return wav_data



    def __getitem__(self, idx):
        wav_data = self.df.iloc[idx]["wav_file"]  
        valence = self.df.iloc[idx]["Valence"]
        arousal = self.df.iloc[idx]["Arousal"]
        
        max_length = self.sample_rate * 10

        inputs = self.processor(wav_data, sampling_rate=self.sample_rate, return_tensors="pt", padding = 'max_length', \
                                 truncation = True, max_length = max_length, do_normalize = True)
        
        input_values = inputs['input_values'].squeeze(0)
        input_values = self.normalize_waveform(input_values)

        """if input_values.min() < -1.0 or input_values.max() > 1.0:
            print(f"Audio not normalized! Min: {input_values.min()}, Max: {input_values.max()}")
        else:
            print(f"Audio normalized! Min: {input_values.min()}, Max: {input_values.max()}")"""

        inputs['input_values'] = input_values
        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)
        return inputs
    
        #check norms and gradients
        #try L1 L2 and batch norm
    

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionClassifier, self).__init__()
        
        # Aggiungo skip connections e batch normalization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h1 = torch.relu(self.norm1(self.fc1(x)))
        h2 = torch.relu(self.norm2(self.fc2(h1)))
        h = h1 + h2  # Skip connection
        out = self.fc3(h)
        return out
    
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(self.config)
        #self.regressor = EmotionClassifier(self.wav2vec2.config.hidden_size, self.wav2vec2.config.hidden_size, 2)
        #self.gradient_checkpointing_enable()
        
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # Allow fine-tuning of transformer layers
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True

        self.regressor = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 2),
            nn.ReLU()
            #nn.BatchNorm1d(self.wav2vec2.config.hidden_size),
            #nn.ReLU(), #nn.Sigmoid()
            #nn.Dropout(self.wav2vec2.config.final_dropout),
            #nn.BatchNorm1d(self.wav2vec2.config.hidden_size),
            #nn.Linear(self.wav2vec2.config.hidden_size, 2)  # Valence and Arousal
        )
        #self.init_weights()

    def forward(
            self,
            input_values,
        ):
        outputs = self.wav2vec2(input_values)
        #print(outputs)
        hidden_states = outputs.last_hidden_state
        
        hidden_states = torch.mean(hidden_states, dim=1)
        
        logits = self.regressor(hidden_states)

        return hidden_states, logits


def plot_distributions(df, columns, title, filename):
    plt.figure(figsize=(12, 6))
    for col in columns:
        sns.histplot(df[col], kde=True, label=col)
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(filename)

def visualize_dataset_distributions(train_df, test_df):
    plot_distributions(train_df, ["Valence", "Arousal"], "Distribuzione - Train Set", "train_distr.png")
    plot_distributions(test_df, ["Valence", "Arousal"], "Distribuzione - Test Set", "test_distr.png")


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


def ccc_func(gold, pred):
    
    #gold = gold.squeeze(-1)
    pred = pred.squeeze(-1)
    
    gold_mean = torch.mean(gold, dim = 0, keepdim=True)
    pred_mean = torch.mean(pred, dim = 0, keepdim=True)

    covariance = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=0)
    gold_var = torch.mean((gold - gold_mean) ** 2, dim=0)
    pred_var = torch.mean((pred - pred_mean) ** 2, dim=0)

    #pearson_coefficient = covariance/(torch.sqrt(gold_var) * torch.sqrt(pred_var))
    
    ccc = (2 * covariance) / (gold_var + pred_var + (gold_mean - pred_mean) ** 2 + 1e-7)
    return ccc


def ccc_loss(gold, pred):
    ccc = ConcordanceCorrCoef().to("cuda")
    coeff = ccc(gold, pred)
    print("CCC:", coeff)
    ccc_loss = 1 - coeff
    return ccc_loss


def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    #attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    return input_values, labels



def compute_loss(model, device, batch, alpha, beta):
    input_values, labels = batch_values(batch, device)

    #TODO Resolve this issue
    if labels[:, 0].std() < 1e-7 or labels[:, 1].std() < 1e-7:
        print("Value equal to 0 or invariance in labels!")
        return None

    _, logits = model(input_values=input_values)#, attention_mask=attention_mask)
    # Example in validation loop
    print("Predictions:", logits[:8].detach().cpu().numpy())
    print("True labels:", labels[:8].detach().cpu().numpy())

    # Compute CCC Loss for valence and arousal
    """def mse(gold, pred):
        return torch.mean((gold-pred)**2)
    loss_val = mse(labels[:, 0], logits[:, 0])
    loss_ar = mse(labels[:, 1], logits[:, 1])"""
    #loss_val = ccc_loss(labels, logits)
    loss_val = ccc_loss(labels[:, 0], logits[:, 0])
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1])

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")
    #print(f"Loss (valence): {loss_val.item()}")
    return loss

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
          epochs=3, alpha=0.5, beta=0.5, checkpoint_path = "model_checkpoint_sampled.pth", patience_es = 8):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    print("****TRAINING****")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    no_improvement_epochs = 0
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=10)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training Loop
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()
           
            loss = compute_loss(model, device, batch, alpha, beta)
            if loss is None: continue 

            # Backpropagation
            print("\tOptimizer lr (before backward): ",optimizer.param_groups[0]['lr'])
            loss.backward()
            print("\tOptimizer lr (after backward): ",optimizer.param_groups[0]['lr'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            log_gradient_norms(model, epoch)
            gradients = get_gradients(model)

            for name, grad in gradients.items():
                grad_norm = np.linalg.norm(grad)
                print(f"Gradient Norm for {name}: {grad_norm}")

            optimizer.step()
            print("\tOptimizer lr (after step): ",optimizer.param_groups[0]['lr'])
            loss = loss.item()
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss}")
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        log_embedding_norms(model, epoch)
        #save_checkpoint(model, optimizer, epoch, checkpoint_path)

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
        scheduler.step(val_loss)

    writer.close()
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

    with torch.no_grad():
        outputs = model(input_values=input_values)
    
    return outputs[1]


def main():
    device = return_device()
    
    pretrained_model = "facebook/wav2vec2-base"    #patrickvonplaten/wav2vec2_tiny_random_robust" #w2v2-L-robust-12
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model, attn_implementation="flash_attention_2")
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    model = EmotionModel(config).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    df = pd.read_pickle("data/MuSe_sample").sample(frac=1, random_state=42).reset_index(drop=True)
    """with open("./data/MuSe_useful", "rb") as f:
            df = pickle.load(f)"""
    df["Valence"] = df["Valence"].clip(0, 1)
    df["Arousal"] = df["Arousal"].clip(0, 1)

    print(df["Valence"].describe())
    print(df["Arousal"].describe())
    
    print(df.head(10))

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    #visualize_dataset_distributions(train_df, test_df)
    #create_block_diagram()
    train_dataset = EmotionDataset(train_df, processor)
    test_dataset = EmotionDataset(test_df, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,\
                                num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle=True,\
                                num_workers=4, pin_memory=True)

    summary(model)
    
    train(model, device, train_dataloader, test_dataloader, epochs = 15)


if __name__ == "__main__":
    main()


#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf
#when the user will test the model, try to:
# - mix the two dataset for training (full_data)
# - train with iemocap and test with muse
# - train with muse and test with iemocap
# - Make in the interface a selector for these three different models and check which is the most useful 
