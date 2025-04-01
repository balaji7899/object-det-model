#!/usr/bin/env python3
"""
evaluation.py

This script evaluates the ACADA model on the source validation set and on three target domain validation sets
(foggy, lowlight, artistic). It computes:
  - Supervised detection loss on the source validation set (as a reference).
  - Domain classification accuracy for the source and each target domain.
  - InfoNCE (contrastive) loss on target domains (by pairing each target batch with a source validation batch).

It also logs detailed metrics to TensorBoard and a log file, and saves sample visualizations.

Usage example:
    python evaluation.py --checkpoint_path checkpoints/best_model.pth \
      --source_val data/source/val \
      --target_foggy_val data/target/foggy/val \
      --target_lowlight_val data/target/lowlight/val \
      --target_artistic_val data/target/artistic/val \
      --batch_size 8
"""

import os
import argparse
import itertools
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import DomainDataset
from model import ACADA
import math

# Set up logging to a file.
logging.basicConfig(filename="evaluation.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

# Define a simple InfoNCE loss (same as in training)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, positives):
        """
        embeddings: Tensor of shape [batch_size, embed_dim]
        positives: Tensor of shape [batch_size, embed_dim]
        Returns the InfoNCE loss.
        """
        batch_size = embeddings.size(0)
        norm_embeddings = nn.functional.normalize(embeddings, dim=1)
        norm_positives = nn.functional.normalize(positives, dim=1)
        logits = torch.mm(norm_embeddings, norm_positives.t()) / self.temperature
        labels = torch.arange(batch_size).to(embeddings.device)
        loss = self.criterion(logits, labels)
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the ACADA model on source and target validation sets.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pth",
                        help="Path to the best model checkpoint (default: checkpoints/best_model.pth)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation (default: 8)")
    parser.add_argument("--source_val", type=str, default="data/source/val",
                        help="Directory for source validation images (default: data/source/val)")
    parser.add_argument("--target_foggy_val", type=str, default="data/target/foggy/val",
                        help="Directory for foggy target validation images (default: data/target/foggy/val)")
    parser.add_argument("--target_lowlight_val", type=str, default="data/target/lowlight/val",
                        help="Directory for low-light target validation images (default: data/target/lowlight/val)")
    parser.add_argument("--target_artistic_val", type=str, default="data/target/artistic/val",
                        help="Directory for artistic target validation images (default: data/target/artistic/val)")
    parser.add_argument("--log_dir", type=str, default="logs_evaluation", help="Directory for evaluation TensorBoard logs (default: logs_evaluation)")
    args = parser.parse_args()
    return args

def evaluate_loader(loader, model, device, detection_criterion, domain_criterion, contrastive_criterion, ground_truth_domain, paired_source_loader=None):
    """
    Evaluate the model on a given loader.
    
    Args:
        loader: DataLoader for the domain (source or target).
        model: ACADA model.
        device: Torch device.
        detection_criterion: Loss function for detection (placeholder).
        domain_criterion: Loss function for domain classification.
        contrastive_criterion: InfoNCE loss.
        ground_truth_domain: int (0 for source, 1 for target).
        paired_source_loader: Optional DataLoader for pairing (for InfoNCE loss).
    
    Returns:
        avg_det_loss: Average detection loss (if applicable).
        domain_acc: Domain classification accuracy.
        avg_contrastive_loss: Average InfoNCE loss (if paired_source_loader is provided, else None).
    """
    model.eval()
    total_det_loss = 0.0
    total_domain_correct = 0
    total_samples = 0
    total_contrastive_loss = 0.0
    contrastive_batches = 0
    
    paired_source_iter = None
    if paired_source_loader is not None:
        paired_source_iter = itertools.cycle(paired_source_loader)
    
    with torch.no_grad():
        for batch in loader:
            images, labels, _ = batch  # labels are dummy for target; for source, they are used for detection loss.
            images = images.to(device)
            labels = labels.to(device)
            
            global_feat, domain_pred, local_emb = model(images)
            
            # For detection loss on source
            det_loss = detection_criterion(global_feat, labels)
            total_det_loss += det_loss.item() * images.size(0)
            
            # Domain classification accuracy
            preds = torch.argmax(domain_pred, dim=1)
            correct = (preds == ground_truth_domain).sum().item()
            total_domain_correct += correct
            total_samples += images.size(0)
            
            # If paired_source_loader is provided, compute InfoNCE loss between target and paired source
            if paired_source_iter is not None:
                source_batch = next(paired_source_iter)
                src_images, _, _ = source_batch
                src_images = src_images.to(device)
                _, _, src_local_emb = model(src_images)
                contrast_loss = contrastive_criterion(local_emb, src_local_emb)
                total_contrastive_loss += contrast_loss.item()
                contrastive_batches += 1
    
    avg_det_loss = total_det_loss / total_samples
    domain_acc = total_domain_correct / total_samples * 100.0
    avg_contrastive_loss = total_contrastive_loss / contrastive_batches if contrastive_batches > 0 else None
    return avg_det_loss, domain_acc, avg_contrastive_loss

def visualize_samples(loader, model, device, save_dir, domain_name, num_samples=4):
    """
    Visualize a few samples with predicted domain labels.
    Saves a figure with images and overlayed predicted domain labels.
    """
    model.eval()
    images_list = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            imgs, _, img_ids = batch
            imgs = imgs.to(device)
            _, domain_pred, _ = model(imgs)
            pred_labels = torch.argmax(domain_pred, dim=1).cpu().numpy()
            for i in range(imgs.size(0)):
                images_list.append(imgs[i].cpu())
                predictions.append((img_ids[i], pred_labels[i]))
            if len(images_list) >= num_samples:
                break

    # Create a figure and plot the images
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img = images_list[i].permute(1, 2, 0).numpy()
        # Denormalize image
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"ID: {predictions[i][0]}\nPred: {predictions[i][1]}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{domain_name}_samples.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved {domain_name} sample visualization to {save_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Data transforms for evaluation (same as training validation)
    transform_val = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets for evaluation
    source_val_dataset = DomainDataset(root_dir=args.source_val, transform=transform_val, domain='source', load_annotations=False)
    source_val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    target_foggy_val_dataset = DomainDataset(root_dir=args.target_foggy_val, transform=transform_val, domain='target')
    target_foggy_val_loader = DataLoader(target_foggy_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    target_lowlight_val_dataset = DomainDataset(root_dir=args.target_lowlight_val, transform=transform_val, domain='target')
    target_lowlight_val_loader = DataLoader(target_lowlight_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    target_artistic_val_dataset = DomainDataset(root_dir=args.target_artistic_val, transform=transform_val, domain='target')
    target_artistic_val_loader = DataLoader(target_artistic_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize the model and load the best checkpoint
    model = ACADA(num_classes=21)
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        logging.info(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        logging.error(f"Checkpoint {args.checkpoint_path} does not exist.")
        return
    model.to(device)
    
    # Define loss functions
    detection_criterion = nn.CrossEntropyLoss()  # Placeholder detection loss
    domain_criterion = nn.CrossEntropyLoss()       # For domain classification
    contrastive_criterion = InfoNCELoss(temperature=0.07)
    
    # Evaluate on source validation set
    src_det_loss, src_domain_acc, _ = evaluate_loader(source_val_loader, model, device,
                                                      detection_criterion, domain_criterion, contrastive_criterion,
                                                      ground_truth_domain=0, paired_source_loader=None)
    logging.info(f"Source Val - Detection Loss: {src_det_loss:.4f}, Domain Accuracy: {src_domain_acc:.2f}%")
    writer.add_scalar("Eval/Source_Detection_Loss", src_det_loss)
    writer.add_scalar("Eval/Source_Domain_Acc", src_domain_acc)
    
    # For target domains, we pair with source validation loader for InfoNCE computation.
    paired_source_loader = source_val_loader  # Use the same source validation loader
    
    # Evaluate target foggy domain
    fog_det_loss, fog_domain_acc, fog_contrast_loss = evaluate_loader(target_foggy_val_loader, model, device,
                                                                      detection_criterion, domain_criterion, contrastive_criterion,
                                                                      ground_truth_domain=1, paired_source_loader=paired_source_loader)
    logging.info(f"Foggy Target Val - Domain Accuracy: {fog_domain_acc:.2f}%, InfoNCE Loss: {fog_contrast_loss:.4f}")
    writer.add_scalar("Eval/Foggy_Domain_Acc", fog_domain_acc)
    writer.add_scalar("Eval/Foggy_InfoNCE_Loss", fog_contrast_loss)
    
    # Evaluate target lowlight domain
    low_det_loss, low_domain_acc, low_contrast_loss = evaluate_loader(target_lowlight_val_loader, model, device,
                                                                      detection_criterion, domain_criterion, contrastive_criterion,
                                                                      ground_truth_domain=1, paired_source_loader=paired_source_loader)
    logging.info(f"Lowlight Target Val - Domain Accuracy: {low_domain_acc:.2f}%, InfoNCE Loss: {low_contrast_loss:.4f}")
    writer.add_scalar("Eval/Lowlight_Domain_Acc", low_domain_acc)
    writer.add_scalar("Eval/Lowlight_InfoNCE_Loss", low_contrast_loss)
    
    # Evaluate target artistic domain
    art_det_loss, art_domain_acc, art_contrast_loss = evaluate_loader(target_artistic_val_loader, model, device,
                                                                      detection_criterion, domain_criterion, contrastive_criterion,
                                                                      ground_truth_domain=1, paired_source_loader=paired_source_loader)
    logging.info(f"Artistic Target Val - Domain Accuracy: {art_domain_acc:.2f}%, InfoNCE Loss: {art_contrast_loss:.4f}")
    writer.add_scalar("Eval/Artistic_Domain_Acc", art_domain_acc)
    writer.add_scalar("Eval/Artistic_InfoNCE_Loss", art_contrast_loss)
    
    # Visualize samples from each evaluation set and save the figures
    vis_dir = "evaluation_visuals"
    visualize_samples(source_val_loader, model, device, vis_dir, "source")
    visualize_samples(target_foggy_val_loader, model, device, vis_dir, "foggy")
    visualize_samples(target_lowlight_val_loader, model, device, vis_dir, "lowlight")
    visualize_samples(target_artistic_val_loader, model, device, vis_dir, "artistic")
    
    writer.close()
    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()
