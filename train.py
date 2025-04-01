#!/usr/bin/env python3
"""
train.py

This script implements the training pipeline for the ACADA model,
which combines supervised detection loss on the source domain with 
domain adversarial and contrastive (InfoNCE) losses. The target domain 
data is loaded via three separate loaders (for foggy, lowlight, and artistic domains)
and alternated in a round-robin fashion.

Usage Example:
    python train.py --batch_size 8 --num_epochs 20 --lr 0.001 \
      --lambda_adv 1.0 --lambda_con 0.1 --lambda_det 1.0
"""

import os
import argparse
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DomainDataset  # Ensure this file is in the project folder
from model import ACADA           # ACADA includes the shared backbone, GRL with dynamic lambda, domain classifier, and projection head

# Define a simple InfoNCE loss (for illustration)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, positives):
        """
        embeddings: Tensor of shape [batch_size, embed_dim]
        positives: Tensor of shape [batch_size, embed_dim] (the positive sample embeddings)
        We compute cosine similarities and use them as logits.
        """
        batch_size = embeddings.size(0)
        # Normalize embeddings
        norm_embeddings = nn.functional.normalize(embeddings, dim=1)
        norm_positives = nn.functional.normalize(positives, dim=1)
        # Compute similarity matrix
        logits = torch.mm(norm_embeddings, norm_positives.t()) / self.temperature
        # Labels: Diagonal entries are positives
        labels = torch.arange(batch_size).to(embeddings.device)
        loss = self.criterion(logits, labels)
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train ACADA model with domain adaptation.")
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Initial weight for adversarial loss (default: 1.0)")
    parser.add_argument("--lambda_con", type=float, default=0.1, help="Weight for contrastive loss (default: 0.1)")
    parser.add_argument("--lambda_det", type=float, default=1.0, help="Weight for detection loss (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss (default: 0.07)")
    # Data directories
    parser.add_argument("--source_train", type=str, default="data/source/train", help="Source train directory")
    parser.add_argument("--source_val", type=str, default="data/source/val", help="Source val directory")
    parser.add_argument("--target_foggy", type=str, default="data/target/foggy", help="Target domain directory for foggy images")
    parser.add_argument("--target_lowlight", type=str, default="data/target/lowlight", help="Target domain directory for low-light images")
    parser.add_argument("--target_artistic", type=str, default="data/target/artistic", help="Target domain directory for artistic images")
    # Checkpoint and logging directories
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for TensorBoard logs")
    args = parser.parse_args()
    return args

def log_total_gradient_norm(model, writer, global_step):
    """Compute and log the total gradient norm for the entire model."""
    total_norm = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            count += 1
    total_norm = total_norm ** 0.5
    writer.add_scalar("GradNorm/Total", total_norm, global_step)

def log_submodule_gradient_norms(model, writer, global_step):
    """Log average gradient norms for key submodules."""
    submodules = {
        "FeatureExtractor": model.feature_extractor,
        "DomainClassifier": model.domain_classifier,
        "ProjectionHead": model.projection_head
    }
    for name, module in submodules.items():
        total_norm = 0.0
        count = 0
        for param in module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item()
                count += 1
        if count > 0:
            avg_norm = total_norm / count
            writer.add_scalar(f"GradNorm/{name}", avg_norm, global_step)

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize model and move to device
    model = ACADA(num_classes=21, lambda_adv=args.lambda_adv).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss functions:
    # Placeholder detection loss on source domain (e.g., classification loss)
    detection_criterion = nn.CrossEntropyLoss()
    # Domain adversarial loss: using cross-entropy
    domain_criterion = nn.CrossEntropyLoss()
    # Contrastive loss (InfoNCE)
    contrastive_criterion = InfoNCELoss(temperature=args.temperature)
    
    # Data transforms
    from torchvision import transforms
    transform_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create source domain loaders (train & val)
    source_train_dataset = DomainDataset(root_dir=args.source_train, transform=transform_train, domain='source', load_annotations=False)
    source_val_dataset   = DomainDataset(root_dir=args.source_val, transform=transform_val, domain='source', load_annotations=False)
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    source_val_loader   = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create separate target loaders for each domain (train)
    target_foggy_train_loader   = DataLoader(DomainDataset(root_dir=os.path.join(args.target_foggy, "train"), transform=transform_train, domain='target'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    target_lowlight_train_loader = DataLoader(DomainDataset(root_dir=os.path.join(args.target_lowlight, "train"), transform=transform_train, domain='target'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    target_artistic_train_loader = DataLoader(DomainDataset(root_dir=os.path.join(args.target_artistic, "train"), transform=transform_train, domain='target'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # For round-robin, combine target loaders into a list
    target_train_loaders = [target_foggy_train_loader, target_lowlight_train_loader, target_artistic_train_loader]
    target_loader_cycle = itertools.cycle(target_train_loaders)
    
    best_val_loss = float('inf')
    global_step = 0
    
    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        # Optionally update GRL lambda dynamically based on training progress.
        # Here, progress is computed as epoch / total epochs (a value between 0 and 1).
        progress = float(epoch) / args.num_epochs
        new_lambda = model.update_lambda(progress, gamma=10)
        writer.add_scalar("Lambda/GRL", new_lambda, epoch+1)
        print(f"Epoch {epoch+1}: Updated GRL lambda to {new_lambda:.4f}")
        
        for source_batch in source_train_loader:
            # Get source batch
            src_images, src_labels, _ = source_batch  # src_labels is a placeholder here
            src_images = src_images.to(device)
            src_labels = src_labels.to(device)
            
            # Get a batch from target domain using round-robin
            try:
                target_batch = next(target_loader_cycle)
            except StopIteration:
                target_loader_cycle = itertools.cycle(target_train_loaders)
                target_batch = next(target_loader_cycle)
            tgt_images, _, _ = target_batch
            tgt_images = tgt_images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for source images
            global_feat_src, domain_pred_src, local_emb_src = model(src_images)
            # Forward pass for target images
            global_feat_tgt, domain_pred_tgt, local_emb_tgt = model(tgt_images)
            
            # Compute supervised detection loss on source
            # Here we use global features and a dummy label (adjust according to your real annotations)
            loss_det = detection_criterion(global_feat_src, src_labels)
            
            # Compute domain adversarial loss:
            # Domain labels: 0 for source, 1 for target
            domain_labels_src = torch.zeros(src_images.size(0), dtype=torch.long).to(device)
            domain_labels_tgt = torch.ones(tgt_images.size(0), dtype=torch.long).to(device)
            loss_domain_src = domain_criterion(domain_pred_src, domain_labels_src)
            loss_domain_tgt = domain_criterion(domain_pred_tgt, domain_labels_tgt)
            loss_domain = loss_domain_src + loss_domain_tgt
            
            # Compute contrastive loss:
            # Here, for simplicity, we use local embeddings from source and target as positives for InfoNCE
            loss_contrastive = contrastive_criterion(local_emb_src, local_emb_tgt)
            
            # Total loss is a weighted sum of the three losses
            total_loss = args.lambda_det * loss_det + args.lambda_adv * loss_domain + args.lambda_con * loss_contrastive
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            # Log per-loss values
            if global_step % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{global_step}], Total Loss: {total_loss.item():.4f}")
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/Detection", loss_det.item(), global_step)
                writer.add_scalar("Loss/Domain", loss_domain.item(), global_step)
                writer.add_scalar("Loss/Contrastive", loss_contrastive.item(), global_step)
                # Log gradient norms
                log_total_gradient_norm(model, writer, global_step)
                log_submodule_gradient_norms(model, writer, global_step)
            global_step += 1
        
        avg_train_loss = running_loss / len(source_train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Epoch/Average_Training_Loss", avg_train_loss, epoch+1)
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Evaluate on source validation set (for demonstration; extend to target val as needed)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in source_val_loader:
                val_images, val_labels, _ = val_batch
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                global_feat_val, domain_pred_val, local_emb_val = model(val_images)
                loss_det_val = detection_criterion(global_feat_val, val_labels)
                val_loss += loss_det_val.item()
            avg_val_loss = val_loss / len(source_val_loader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Val/Loss", avg_val_loss, epoch+1)
            
            # Save best model checkpoint based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Best model updated at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
    
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
