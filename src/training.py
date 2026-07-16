import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pytorch_metric_learning import losses
from src.dataset import PigReIDDataset
from src.reid_model import PigReIDModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_reid_model(
    train_df,
    train_transform,
    num_classes,
    device,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    embedding_dim=256,
    seed=42
):
    
    set_seed(seed)
    
    train_dataset = PigReIDDataset(
        dataframe=train_df,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    model = PigReIDModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=True
    )
    
    model = model.to(device)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    
    triplet_loss_fn = losses.TripletMarginLoss(margin=0.3)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate
    )
    
    train_history = []
    
    for epoch in range(epochs):
        model.train()
        
        total_loss_sum = 0.0
        
        ce_loss_sum = 0.0
        
        triplet_loss_sum = 0.0
        
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            
            images = batch["image"]
            
            labels = batch["label"]
            
            images = images.to(device)
            
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            embeddings, logits = model(images)
            
            loss_ce = ce_loss_fn(logits, labels)
            
            loss_triplet = triplet_loss_fn(embeddings, labels)
            
            loss = loss_ce + loss_triplet
            
            loss.backward()
            
            optimizer.step()
            
            total_loss_sum += loss.item()
            
            ce_loss_sum += loss_ce.item()
            
            triplet_loss_sum += loss_triplet.item()
            
            num_batches += 1
        
        avg_total_loss = total_loss_sum / num_batches
        
        avg_ce_loss = ce_loss_sum / num_batches
        
        avg_triplet_loss = triplet_loss_sum / num_batches
        
        train_history.append({
            "epoch": epoch + 1,
            "total_loss": avg_total_loss,
            "ce_loss": avg_ce_loss,
            "triplet_loss": avg_triplet_loss
        })
        
        print(
            f"Epoch {epoch + 1}: "
            f"total_loss={avg_total_loss:.4f}, "
            f"ce_loss={avg_ce_loss:.4f}, "
            f"triplet_loss={avg_triplet_loss:.4f}"
        )
    
    return model, train_history