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


class EmotionDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.sample_rate = 16000
        self.max_seconds = 6  #max padding seconds
        self.threshold = 0.8  #max percentage of which files to keep


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
        
        max_length = self.sample_rate * self.max_seconds

        if len(wav_data) > max_length / self.threshold:
            return self.__getitem__((idx + 1) % len(self.df))
        
        inputs = self.processor(wav_data, sampling_rate=self.sample_rate, return_tensors="pt", padding = 'max_length', \
                                truncation = True, max_length = max_length, do_normalize = True)
        
        input_values = inputs['input_values'].squeeze(0)
        input_values = self.normalize_waveform(input_values)

        inputs['input_values'] = input_values
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

        """self.rnn = nn.LSTM(input_size= self.wav2vec2.config.hidden_size, hidden_size=self.wav2vec2.config.hidden_size,\
                           batch_first=True, bidirectional=False)"""
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.wav2vec2.config.hidden_size, 128), 
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

        self.init_weights()

    def forward(
            self,
            input_values,
        ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.mean(hidden_states, dim=1)
        #hidden_states,_ = self.rnn(hidden_states)
        logits = self.regressor(hidden_states)

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
    #attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    return input_values, labels



def compute_loss(model, device, batch, alpha, beta):
    input_values, labels = batch_values(batch, device)

    #For small batch sizes where variance could be low
    if labels[:, 0].std() < 1e-7 or labels[:, 1].std() < 1e-7:
        print("Value equal to 0 or invariance in labels!")
        return None

    _, logits = model(input_values=input_values)#, attention_mask=attention_mask)
    # Example in validation loop
    print("Predictions:", logits[:8].detach().cpu().numpy())
    print("True labels:", labels[:8].detach().cpu().numpy())

    loss_val = ccc_loss(labels[:, 0], logits[:, 0])
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1])

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")
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
          epochs=3, alpha=0.5, beta=0.5, checkpoint_path = "model_checkpoint_sampled.pth", patience_es = 15):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    print("****TRAINING****")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    no_improvement_epochs = 0
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(train_dataloader), epochs=10)

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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def plot_losses(train_losses, val_losses, filename = "./loss_plot_trial.png"):
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

    df = pd.read_pickle("data/MuSe_sample").sample(frac=0.6, random_state=42).reset_index(drop=True)

    print(df["Valence"].describe())
    print(df["Arousal"].describe())
    
    print(df.head(10))

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    train_dataset = EmotionDataset(train_df, processor)
    test_dataset = EmotionDataset(test_df, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True,\
                                num_workers=4, pin_memory=True, drop_last = True)

    summary(model)
    
    train(model, device, train_dataloader, test_dataloader, epochs = 25)


if __name__ == "__main__":
    main()


#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf
#when the user will test the model, try to:
# - mix the two dataset for training (full_data)
# - train with iemocap and test with muse
# - train with muse and test with iemocap
# - Make in the interface a selector for these three different models and check which is the most useful 
