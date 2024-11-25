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

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)

from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeBlock

from torch.distributed import init_process_group, all_reduce
from torchinfo import summary
import datetime


# Initialize Distributed Process Group for FSDP
def initialize_distributed():   
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=60))
    

def requires_grad_policy(module, recurse, *args, **kwargs):
    """Wrap only modules that have trainable parameters."""
    return any(p.requires_grad for p in module.parameters())

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
        
        # Process audio input
        inputs = self.processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs['labels'] = torch.tensor([valence, arousal], dtype=torch.float32)
        
        return inputs
    

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        #x = self.dropout(x)
        #x = self.dense(x)
        x = torch.tanh(x)        
        #x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
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

def distributed_loss(tensor, world):
    return all_reduce(tensor, op=dist.ReduceOp.SUM) / world

def batch_values(batch, device):
    input_values = batch['input_values'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    return input_values, attention_mask, labels


def compute_loss(model, device, batch, alpha, beta):
    input_values, attention_mask, labels = batch_values(batch, device)
    
    #TODO Resolve this issue
    if torch.any(torch.eq(labels,0)) or (labels[:, 0].var() == 0 or labels[:, 1].var() == 0):
        print("Value equal to 0 or invariance in labels!")
        return None

    _, logits = model(input_values=input_values, attention_mask=attention_mask)
    print("logits:", logits)
    print("labels:", labels)
    
    # Compute CCC Loss for valence and arousal
    loss_val = ccc_loss(labels[:, 0], logits[:, 0]).mean()
    loss_ar = ccc_loss(labels[:, 1], logits[:, 1]).mean()

    # Weighted total loss
    loss = alpha * loss_val + beta * loss_ar
    print(f"Loss (valence): {loss_val.item()}, Loss (arousal): {loss_ar.item()}, Total: {loss.item()}")

    return loss

def train_cpu(model, train_dataloader, test_dataloader, optimizer, epochs, alpha, beta):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    model.train()
    checkpoint_path = "model_checkpoint_sampled.pth"
    print("****TRAINING****")
    for epoch in range(epochs):
        epoch_loss = 0

        # Training Loop
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            loss = compute_loss(model, 'cpu', batch, alpha, beta)
            if loss is None: continue 

            # Backpropagation
            print("before backpropagation")
            loss.backward()
            print("after backpropagation")

            optimizer.step()
            #loss = loss.detach()
            # avg loss over all processes
            epoch_loss += loss

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(train_dataloader)}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Validation Loop
        validate(model, test_dataloader, alpha, beta)
        #TODO undersand why in the last step there is a mismatch between batch and input

def train_gpu(model, train_dataloader, test_dataloader, optimizer, epochs, alpha, beta):
    """
    Train the model using CCC loss for valence and arousal.
    """
    #TODO try to refactor with Accelerate (HuggingFace)
    dist.barrier()
    model.train()
    checkpoint_path = "model_checkpoint_sampled.pth"
    print("****TRAINING****")
    for epoch in range(epochs):
        epoch_loss = 0

        # Training Loop
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            loss = compute_loss(model, 'cuda', batch, alpha, beta)
            if loss is None: continue 

            # Backpropagation
            print("before backpropagation")
            loss.backward()
            print("after backpropagation")

            optimizer.step()
            #loss = loss.detach()
            dist.barrier()
            # avg loss over all processes
            """loss_tensor = loss.clone().detach()
            print("Loss Tensor:",loss_tensor)
            loss = distributed_loss(loss_tensor, float(dist.get_world_size())).item()
            print("reduction loss:", loss)"""
            epoch_loss += loss
            print("Allocated:", torch.cuda.memory_allocated(), "Reserved:", torch.cuda.memory_reserved())


        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(train_dataloader)}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Validation Loop
        validate(model, test_dataloader, alpha, beta)
        #TODO undersand why in the last step there is a mismatch between batch and input

def train(model, device, train_dataloader, test_dataloader, optimizer, epochs=3, alpha=0.5, beta=0.5):
    if device == 'cpu':
        train_cpu(model, train_dataloader, test_dataloader, optimizer, epochs=3, alpha=0.5, beta=0.5)
    else: train_gpu(model, train_dataloader, test_dataloader, optimizer, epochs=3, alpha=0.5, beta=0.5)


# Validation Loop
def validate(model, test_dataloader, alpha, beta):
    model.eval()
    val_loss = 0
    print("****VALIDATION****")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss = compute_loss(model, batch, alpha, beta)
            if loss is None: continue
            print(loss)
            val_loss += loss.item()

    # Average CCC scores
    print(f"Validation Loss: {val_loss / len(test_dataloader)}")


def load_trained_model(device, checkpoint_path, pretrained_model):
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


def predict_emotion(model, device, processor, wav_data):
    model.eval()
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].to(device)
    attention_mask = inputs.get('attention_mask').to(device) if 'attention_mask' in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
    
    return outputs[1]


def main():
    initialize_distributed()

    device = 'cpu' #return_device()
    pretrained_model = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    config = Wav2Vec2Config.from_pretrained(pretrained_model)
    config.num_labels = 2  # Ensure this matches the number of regression outputs (Valence, Arousal)  
    model = EmotionModel(config).to(device)
    model.gradient_checkpointing_enable()

    df = pd.read_pickle("data/full_data")
    df_sampled = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    print(df_sampled.head())

    train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=42)

    train_dataset = EmotionDataset(train_df, processor)
    test_dataset = EmotionDataset(test_df, processor)


    if device!='cpu':
        model = FSDP(model, auto_wrap_policy=requires_grad_policy, use_orig_params=True, sync_module_states=True,\
                 sharding_strategy=ShardingStrategy.FULL_SHARD, backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
                 #,cpu_offload=CPUOffload(offload_params=True))
        
        train_dataloader = DataLoader(train_dataset, batch_size=2, sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),\
                                 collate_fn=custom_collate, \
                                 num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=2, sampler=DistributedSampler(test_dataset, shuffle=True, drop_last=True),   
                                 collate_fn=custom_collate, \
                                 num_workers=4, pin_memory=True)  


    print(summary(model))
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,\
                                 collate_fn=custom_collate, \
                                 num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2,shuffle=True,\
                                 collate_fn=custom_collate, \
                                 num_workers=4, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train(model, device, train_dataloader, test_dataloader, optimizer, epochs = 5)



#if __name__ == "__main ":
main()



#remember to do a scatterplot for valence and arousal like paper https://iopscience.iop.org/article/10.1088/1742-6596/1896/1/012004/pdf
#when the user will test the model, try to:
# - mix the two dataset for training (full_data)
# - train with iemocap and test with muse
# - train with muse and test with iemocap
# - Make in the interface a selector for these three different models and check which is the most useful 


#trovare un modo per spezzare i gradienti

#pipeline parallelism naive
#torch.cuda.memory_allocated
#change to float32

#For execute it: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 vocal_assistant/emotion/predict_emotion.py
