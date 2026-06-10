import os
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    ACDCH5SliceDataset,
    split_patients_from_slices,
    build_client_subsets_from_indices,
)
from multitask_model import FederatedMultiTaskModel


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = torch.sum(probs * onehot, dims)
    union = torch.sum(probs + onehot, dims)

    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def get_weights(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def set_weights(model, weights):
    model.load_state_dict(weights, strict=True)


def fedavg(client_weights, sizes):
    total = float(sum(sizes))
    avg = OrderedDict()

    for k in client_weights[0].keys():
        avg[k] = sum(w[k] * (n / total) for w, n in zip(client_weights, sizes))

    return avg


def train_local(global_weights, loader):
    model = FederatedMultiTaskModel(
        num_seg_classes=Config.NUM_SEG_CLASSES,
        num_disease_classes=5
    ).to(Config.DEVICE)

    set_weights(model, global_weights)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    model.train()

    for _ in range(Config.LOCAL_EPOCHS):
        for imgs, masks, disease_labels, _ in loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            disease_labels = disease_labels.to(Config.DEVICE)

            opt.zero_grad()

            outputs = model(imgs)
            seg_logits = outputs["segmentation"]
            disease_logits = outputs["disease"]

            seg_ce = F.cross_entropy(seg_logits, masks)
            seg_dice = dice_loss(seg_logits, masks)
            cls_loss = F.cross_entropy(disease_logits, disease_labels)

            loss = 0.4 * seg_ce + 0.4 * seg_dice + 0.2 * cls_loss
            loss.backward()
            opt.step()

    return get_weights(model)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, masks, disease_labels, _ in loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            disease_labels = disease_labels.to(Config.DEVICE)

            outputs = model(imgs)
            seg_logits = outputs["segmentation"]
            disease_logits = outputs["disease"]

            seg_ce = F.cross_entropy(seg_logits, masks)
            cls_loss = F.cross_entropy(disease_logits, disease_labels)
            loss = seg_ce + cls_loss

            total_loss += loss.item()

            preds = torch.argmax(disease_logits, dim=1)
            correct += (preds == disease_labels).sum().item()
            total += disease_labels.size(0)

    acc = correct / total if total > 0 else 0.0
    return total_loss, acc


def main():
    seed_all(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    slices_dir = Config.PREPROCESSED_SLICES_DIR
    if not os.path.isdir(slices_dir):
        raise FileNotFoundError(f"Could not find slices dir: {slices_dir}")

    train_indices, val_indices = split_patients_from_slices(
        slices_dir,
        Config.VAL_RATIO,
        Config.SEED
    )

    val_dataset = ACDCH5SliceDataset(
        slices_dir=slices_dir,
        indices=val_indices,
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    client_datasets = build_client_subsets_from_indices(
        slices_dir,
        train_indices,
        Config.NUM_CLIENTS,
        Config.SEED
    )

    client_loaders = {
        cid: DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        for cid, ds in client_datasets.items()
    }

    global_model = FederatedMultiTaskModel(
        num_seg_classes=Config.NUM_SEG_CLASSES,
        num_disease_classes=5
    ).to(Config.DEVICE)

    global_weights = get_weights(global_model)
    best_val_loss = None

    print(f"Using slices dir: {slices_dir}")
    print(f"Validation samples: {len(val_dataset)}")
    for cid, ds in client_datasets.items():
        print(f"Client {cid}: {len(ds)} samples")

    for rnd in range(Config.GLOBAL_ROUNDS):
        print(f"\nRound {rnd + 1}/{Config.GLOBAL_ROUNDS}")

        client_weights = []
        sizes = []

        for cid, loader in client_loaders.items():
            print(f"Client {cid} training...")
            w = train_local(global_weights, loader)
            client_weights.append(w)
            sizes.append(len(loader.dataset))

        global_weights = fedavg(client_weights, sizes)
        set_weights(global_model, global_weights)

        val_loss, val_acc = evaluate(global_model, val_loader)
        print("Validation Loss:", val_loss)
        print("Validation Disease Accuracy:", val_acc)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(global_weights, os.path.join(Config.OUTPUT_DIR, "fed_multitask_best.pt"))
            print("Best multitask model saved.")

    print("Multi-task Training Done!")


if __name__ == "__main__":
    main()