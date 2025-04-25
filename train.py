# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)
from torch.cuda.amp import autocast, GradScaler
from model import Model  # Custom U-Net model

# --- Utility Functions and Mappings ---

# Mapping Cityscapes original class IDs to training IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

# Convert label image from raw IDs to training IDs
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Define mapping from training IDs to colors (used for visualization)
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

# Convert predicted labels to RGB color image for visualization
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

# --- Evaluation Metrics ---

# Compute per-class IoU and return mean IoU
def compute_iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = 19, ignore_index: int = 255):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        if ignore_index is not None:
            ignore_mask = (target == ignore_index)
            pred_inds = np.logical_and(~ignore_mask, pred_inds)
            target_inds = np.logical_and(~ignore_mask, target_inds)

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Compute per-class Dice score and return mean Dice
def compute_dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = 19, ignore_index: int = 255):
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        if ignore_index is not None:
            ignore_mask = (target == ignore_index)
            pred_inds = np.logical_and(~ignore_mask, pred_inds)
            target_inds = np.logical_and(~ignore_mask, target_inds)

        intersection = np.logical_and(pred_inds, target_inds).sum()
        denom = pred_inds.sum() + target_inds.sum()
        if denom == 0:
            dices.append(np.nan)
        else:
            dices.append(2.0 * intersection / denom)
    return np.nanmean(dices)

def combined_loss(outputs, labels, ce_weight=0.7, dice_weight=0.3):
    labels = labels.clone()
    labels[labels == 255] = 0  # prevent one_hot from crashing on ignore_index
    ce = nn.CrossEntropyLoss(ignore_index=255)(outputs, labels)
    pred = torch.softmax(outputs, dim=1)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=19).permute(0, 3, 1, 2).float()
    labels_onehot = labels_onehot.to(pred.device)
    dims = (0, 2, 3)
    intersection = torch.sum(pred * labels_onehot, dims)
    union = torch.sum(pred + labels_onehot, dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1 - dice.mean()
    return ce_weight * ce + dice_weight * dice_loss

# --- Argument Parser ---

def get_args_parser():
    parser = ArgumentParser("Training script for U-Net model (no augmentation, with IoU/Dice logging)")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="unet-noaug")
    return parser

# --- Main Training Loop ---

def main(args):
    # Initialize Weights & Biases for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation-Final",
        name=args.experiment_id,
        config=vars(args),
    )

    # Create checkpoint directory
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seeds and device
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define preprocessing transforms
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # Load datasets
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)

    # Wrap datasets for transforms
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model, loss, optimizer, scheduler, and scaler
    model = Model(in_channels=3, n_classes=19, deep_supervision=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8, verbose=True, min_lr=1e-6)
    scaler = GradScaler()

    # Tracking best metrics
    best_valid_loss = float('inf')
    best_dice_score = 0.0
    best_dice_model_path = None
    best_dice_epoch = -1
    best_loss_model_path = None

    # --- Training Loop ---
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                #loss = criterion(outputs, labels)
                loss = combined_loss(outputs, labels, ce_weight=0.5, dice_weight=0.5)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log training stats
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_loader) + i)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            losses = []
            iou_scores = []
            dice_scores = []
            for i, (images, labels) in enumerate(valid_loader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                #loss = criterion(outputs, labels)
                loss = combined_loss(outputs, labels, ce_weight=0.5, dice_weight=0.5)
                losses.append(loss.item())

                preds = outputs.softmax(1).argmax(1)
                for p, l in zip(preds, labels):
                    iou_scores.append(compute_iou_per_class(p.cpu().numpy(), l.cpu().numpy()))
                    dice_scores.append(compute_dice_per_class(p.cpu().numpy(), l.cpu().numpy()))

                # Visualize first batch
                if i == 0:
                    vis_preds = convert_train_id_to_color(preds.unsqueeze(1))
                    vis_labels = convert_train_id_to_color(labels.unsqueeze(1))
                    wandb.log({
                        "predictions": [wandb.Image(make_grid(vis_preds.cpu(), nrow=8).permute(1, 2, 0).numpy())],
                        "labels": [wandb.Image(make_grid(vis_labels.cpu(), nrow=8).permute(1, 2, 0).numpy())],
                    }, step=(epoch + 1) * len(train_loader) - 1)

            # Compute average metrics
            valid_loss = sum(losses) / len(losses)
            mean_iou = sum(iou_scores) / len(iou_scores)
            mean_dice = sum(dice_scores) / len(dice_scores)

            # Adjust learning rate
            scheduler.step(valid_loss)

            # Log validation stats
            wandb.log({
                "valid_loss": valid_loss,
                "mean_IoU": mean_iou,
                "mean_Dice": mean_dice,
            }, step=(epoch + 1) * len(train_loader) - 1)

            # Save best models
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_loss_model_path:
                    os.remove(best_loss_model_path)
                best_loss_model_path = os.path.join(
                    output_dir, f"best_model_loss-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth"
                )
                torch.save(model.state_dict(), best_loss_model_path)

            if mean_dice > best_dice_score:
                best_dice_score = mean_dice
                best_dice_epoch = epoch + 1
                if best_dice_model_path:
                    os.remove(best_dice_model_path)
                best_dice_model_path = os.path.join(
                    output_dir, f"best_model_dice-epoch={epoch:04}-mean_dice={mean_dice:.4f}.pth"
                )
                torch.save(model.state_dict(), best_dice_model_path)

    # --- Final Save ---
    print(f"Best Mean Dice: {best_dice_score:.4f} at Epoch {best_dice_epoch}")
    print("Training complete.")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth"))
    wandb.finish()

# --- Entry Point ---
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
