import os
import random
import csv
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from config import Config
from dataset import (
    ACDCH5SliceDataset,
    split_patients_from_slices,
    build_client_subsets_from_indices,
)
from deeplab_model import build_deeplab


# =========================================
# Utilities
# =========================================
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


def ensure_single_channel(imgs):
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)
    elif imgs.dim() == 4 and imgs.shape[1] == 3:
        imgs = imgs.mean(dim=1, keepdim=True)
    elif imgs.dim() == 4 and imgs.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected input shape: {imgs.shape}")
    return imgs


def ensure_single_channel_tensor(img):
    if img.dim() == 2:
        return img.unsqueeze(0)
    elif img.dim() == 3 and img.shape[0] == 3:
        return img.mean(dim=0, keepdim=True)
    elif img.dim() == 3 and img.shape[0] == 1:
        return img
    else:
        raise ValueError(f"Unexpected tensor shape: {img.shape}")


def disease_label_to_int(label):
    """
    Converts ACDC disease labels to integers.
    """
    mapping = {
        "NOR": 0,
        "DCM": 1,
        "HCM": 2,
        "MINF": 3,
        "RV": 4
    }

    if torch.is_tensor(label):
        return label.long()

    if isinstance(label, (list, tuple)):
        return torch.tensor([mapping.get(str(x), 0) for x in label], dtype=torch.long)

    if isinstance(label, str):
        return torch.tensor(mapping.get(label, 0), dtype=torch.long)

    return torch.tensor(label, dtype=torch.long)


def extract_disease_labels_from_batch(batch, imgs):
    """
    Extracts disease labels from dataset batch.
    If labels are strings, converts them to numeric classes.
    """
    if len(batch) >= 4:
        labels = batch[3]

        if torch.is_tensor(labels):
            return labels.long()

        if isinstance(labels, (list, tuple)):
            return torch.tensor(
                [disease_label_to_int(x).item() for x in labels],
                dtype=torch.long
            )

        return disease_label_to_int(labels).view(-1)

    return torch.zeros(imgs.size(0), dtype=torch.long)

    """
    Tries to extract disease labels from dataset output.
    Expected batch formats may be:
    (imgs, masks, patient_id, disease_label)
    or other formats.

    If no disease label exists, labels default to 0.
    """
    if len(batch) >= 4:
        labels = batch[3]
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)
        return labels.long()

    return torch.zeros(imgs.size(0), dtype=torch.long)


def extract_single_disease_label(rest):
    """
    Extracts one disease label from a single dataset item.
    If not available, returns 0.
    """
    mapping = {
        "NOR": 0,
        "DCM": 1,
        "HCM": 2,
        "MINF": 3,
        "RV": 4
    }

    if len(rest) >= 2:
        label = rest[1]

        if torch.is_tensor(label):
            return int(label.item())

        if isinstance(label, str):
            return mapping.get(label, 0)

        return int(label)

    return 0
    
    """
    Extracts one disease label from a single dataset item.
    If not available, returns 0.
    """
    if len(rest) >= 2:
        label = rest[1]
        if torch.is_tensor(label):
            return int(label.item())
        return int(label)

    return 0


# =========================================
# Local ACGAN
# =========================================
class SimpleGenerator(nn.Module):
    """
    ACGAN Generator:
    Takes random noise vector z + disease class label.
    """
    def __init__(self, z_dim=64, num_classes=5, img_size=256, embed_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.fc = nn.Linear(z_dim + embed_dim, 128 * 16 * 16)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 64 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),     # 128 -> 256
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([z, label_vec], dim=1)
        x = self.fc(x).view(z.size(0), 128, 16, 16)
        return self.net(x)


class SimpleDiscriminator(nn.Module):
    """
    ACGAN Discriminator:
    Outputs:
    1. validity score: real/fake
    2. class logits: disease class prediction
    """
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),   # 256 -> 128
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 32, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.adv_head = nn.Linear(128, 1)
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.features(x).flatten(1)
        validity = self.adv_head(f)
        class_logits = self.cls_head(f)
        return validity, class_logits


def train_local_gan(loader, client_id, round_id, save_dir, z_dim=64, gan_epochs=1):
    """
    Trains a local ACGAN on client images only.
    Returns trained generator.
    """
    G = SimpleGenerator(
        z_dim=z_dim,
        num_classes=Config.NUM_DISEASE_CLASSES
    ).to(Config.DEVICE)

    D = SimpleDiscriminator(
        num_classes=Config.NUM_DISEASE_CLASSES
    ).to(Config.DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    G.train()
    D.train()

    for ep in range(gan_epochs):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        d_adv_sum = 0.0
        d_cls_sum = 0.0
        g_adv_sum = 0.0
        g_cls_sum = 0.0
        num_batches = 0

        for batch in loader:
            imgs = batch[0]
            labels = extract_disease_labels_from_batch(batch, imgs)

            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).long()

            imgs = ensure_single_channel(imgs)
            real = imgs * 2.0 - 1.0

            n = real.size(0)
            z = torch.randn(n, z_dim, device=Config.DEVICE)
            fake = G(z, labels)

            ones = torch.ones((n, 1), device=Config.DEVICE)
            zeros = torch.zeros((n, 1), device=Config.DEVICE)

            # -------------------------
            # Train Discriminator
            # -------------------------
            real_validity, real_cls = D(real)
            fake_validity, fake_cls = D(fake.detach())

            d_adv_loss = bce(real_validity, ones) + bce(fake_validity, zeros)
            d_cls_loss = ce(real_cls, labels)

            d_loss = d_adv_loss + d_cls_loss

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # -------------------------
            # Train Generator
            # -------------------------
            fake_validity, fake_cls = D(fake)

            g_adv_loss = bce(fake_validity, ones)
            g_cls_loss = ce(fake_cls, labels)

            g_loss = g_adv_loss + g_cls_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_adv_sum += d_adv_loss.item()
            d_cls_sum += d_cls_loss.item()
            g_adv_sum += g_adv_loss.item()
            g_cls_sum += g_cls_loss.item()
            num_batches += 1

        print(
            f"[ACGAN] Client {client_id} | Round {round_id} | "
            f"Epoch {ep+1}/{gan_epochs} | "
            f"D_loss={d_loss_sum/max(1, num_batches):.4f} | "
            f"G_loss={g_loss_sum/max(1, num_batches):.4f} | "
            f"D_adv={d_adv_sum/max(1, num_batches):.4f} | "
            f"D_cls={d_cls_sum/max(1, num_batches):.4f} | "
            f"G_adv={g_adv_sum/max(1, num_batches):.4f} | "
            f"G_cls={g_cls_sum/max(1, num_batches):.4f}"
        )

    os.makedirs(save_dir, exist_ok=True)
    gan_path = os.path.join(save_dir, f"client_{client_id}_round_{round_id}_acgan.pt")
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, gan_path)
    print(f"[ACGAN] Saved local ACGAN: {gan_path}")

    return G


# =========================================
# Augmented Local Dataset
# =========================================
class MixedAugmentedDataset(Dataset):
    """
    Mixes:
    - original real samples
    - synthetic image samples generated locally by ACGAN

    For simplicity, each synthetic image reuses a randomly selected real mask.
    The synthetic image is conditioned on the disease label of that selected real sample.
    """
    def __init__(self, base_dataset, generator, num_synthetic=100, z_dim=64):
        self.base_dataset = base_dataset
        self.generator = generator
        self.num_synthetic = num_synthetic
        self.z_dim = z_dim

        self.synthetic_cache = []
        self._build_synthetic_cache()

    def _build_synthetic_cache(self):
        self.generator.eval()
        self.synthetic_cache = []

        with torch.no_grad():
            for _ in range(self.num_synthetic):
                idx = random.randint(0, len(self.base_dataset) - 1)
                real_img, real_mask, *rest = self.base_dataset[idx]

                disease_label = extract_single_disease_label(rest)
                label_tensor = torch.tensor(
                    [disease_label],
                    device=Config.DEVICE,
                    dtype=torch.long
                )

                z = torch.randn(1, self.z_dim, device=Config.DEVICE)

                fake_img = self.generator(z, label_tensor).squeeze(0).cpu()
                fake_img = (fake_img + 1.0) / 2.0
                fake_img = torch.clamp(fake_img, 0.0, 1.0)

                fake_img = ensure_single_channel_tensor(fake_img)
                self.synthetic_cache.append((fake_img, real_mask.clone()))

    def __len__(self):
        return len(self.base_dataset) + len(self.synthetic_cache)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            item = self.base_dataset[idx]
            img, mask = item[0], item[1]

            img = ensure_single_channel_tensor(img)

            return img.float(), mask.long()

        sidx = idx - len(self.base_dataset)
        img, mask = self.synthetic_cache[sidx]

        img = ensure_single_channel_tensor(img)

        return img.float(), mask.long()


# =========================================
# Local DeepLab training
# =========================================
def train_local(global_weights, client_dataset, client_id, round_id):
    model = build_deeplab(num_seg_classes=Config.NUM_SEG_CLASSES).to(Config.DEVICE)
    set_weights(model, global_weights)

    real_loader = DataLoader(
        client_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    # -------------------------
    # 1) Train local ACGAN
    # -------------------------
    gan_dir = os.path.join(Config.OUTPUT_DIR, "local_acgans")
    G = train_local_gan(
        loader=real_loader,
        client_id=client_id,
        round_id=round_id,
        save_dir=gan_dir,
        z_dim=64,
        gan_epochs=1
    )

    # -------------------------
    # 2) Build mixed dataset
    # -------------------------
    synthetic_count = min(50, len(client_dataset))
    mixed_dataset = MixedAugmentedDataset(
        base_dataset=client_dataset,
        generator=G,
        num_synthetic=synthetic_count,
        z_dim=64
    )

    mixed_loader = DataLoader(
        mixed_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    print(
        f"[DeepLab] Client {client_id} | Round {round_id} | "
        f"Real={len(client_dataset)} | Synthetic={synthetic_count} | "
        f"Total={len(mixed_dataset)}"
    )

    # -------------------------
    # 3) Train local DeepLab
    # -------------------------
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    model.train()

    for ep in range(Config.LOCAL_EPOCHS):
        loss_sum = 0.0
        num_batches = 0

        for imgs, masks in mixed_loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            imgs = ensure_single_channel(imgs)

            opt.zero_grad()
            outputs = model(imgs)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs

            loss = 0.5 * F.cross_entropy(logits, masks) + 0.5 * dice_loss(logits, masks)

            loss.backward()
            opt.step()

            loss_sum += loss.item()
            num_batches += 1

        print(
            f"[DeepLab] Client {client_id} | Round {round_id} | "
            f"Epoch {ep+1}/{Config.LOCAL_EPOCHS} | "
            f"Train Loss={loss_sum/max(1, num_batches):.4f}"
        )

    return get_weights(model), len(mixed_dataset)


# =========================================
# Evaluation
# =========================================
def compute_segmentation_metrics(logits, masks, num_classes, eps=1e-6):
    preds = torch.argmax(logits, dim=1)

    pixel_acc = (preds == masks).float().mean().item()

    dice_scores = []
    iou_scores = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()

        intersection = (pred_cls * mask_cls).sum()
        pred_sum = pred_cls.sum()
        mask_sum = mask_cls.sum()
        union = pred_sum + mask_sum - intersection

        if mask_sum.item() == 0 and pred_sum.item() == 0:
            continue

        dice = (2.0 * intersection + eps) / (pred_sum + mask_sum + eps)
        iou = (intersection + eps) / (union + eps)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    return pixel_acc, mean_dice, mean_iou


def evaluate_model(model, loader):
    model.eval()

    total_loss = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks, _, _ in loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            imgs = ensure_single_channel(imgs)

            outputs = model(imgs)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs

            loss = F.cross_entropy(logits, masks)
            pixel_acc, mean_dice, mean_iou = compute_segmentation_metrics(
                logits, masks, Config.NUM_SEG_CLASSES
            )

            total_loss += loss.item()
            total_pixel_acc += pixel_acc
            total_dice += mean_dice
            total_iou += mean_iou
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / num_batches,
        total_pixel_acc / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


# =========================================
# Main
# =========================================
def main():
    seed_all(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"[PATH] Current working directory: {os.getcwd()}")
    print(f"[PATH] Output dir: {os.path.abspath(Config.OUTPUT_DIR)}")
    print(f"[PATH] Best model path: {os.path.abspath(Config.BEST_MODEL_PATH)}")

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

    global_model = build_deeplab(num_seg_classes=Config.NUM_SEG_CLASSES).to(Config.DEVICE)
    global_weights = get_weights(global_model)

    best_mean_dice = -1.0
    results_log = []

    print(f"Using slices dir: {slices_dir}")
    print(f"Validation samples: {len(val_dataset)}")

    for cid, ds in client_datasets.items():
        print(f"Client {cid}: {len(ds)} samples")

    for rnd in range(Config.GLOBAL_ROUNDS):
        print("\n" + "=" * 70)
        print(f"Federated Round {rnd + 1}/{Config.GLOBAL_ROUNDS}")
        print("=" * 70)

        client_weights = []
        sizes = []

        for cid, ds in client_datasets.items():
            print(f"\n[INFO] Client {cid} local ACGAN + DeepLab pipeline started...")
            w, effective_size = train_local(global_weights, ds, cid, rnd + 1)
            client_weights.append(w)
            sizes.append(effective_size)
            print(f"[INFO] Client {cid} local pipeline finished.")

        print("\n[INFO] Aggregating client weights with FedAvg...")
        global_weights = fedavg(client_weights, sizes)
        set_weights(global_model, global_weights)

        print("[INFO] Evaluating global model on validation set...")
        val_loss, pixel_acc, mean_dice, mean_iou = evaluate_model(global_model, val_loader)

        results_log.append({
            "round": rnd + 1,
            "val_loss": val_loss,
            "pixel_accuracy": pixel_acc,
            "mean_dice": mean_dice,
            "mean_iou": mean_iou
        })

        # Notebook-style output after each round
        print("\nCurrent Training and Validation Summary")
        print("-" * 60)
        print(f"Round              : {rnd + 1}")
        print(f"Validation Loss    : {val_loss:.6f}")
        print(f"Pixel Accuracy     : {pixel_acc:.6f}")
        print(f"Mean Dice Score    : {mean_dice:.6f}")
        print(f"Mean IoU Score     : {mean_iou:.6f}")
        print("-" * 60)

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(global_weights, Config.BEST_MODEL_PATH)
            print(f"[SAVE] Best global model saved to: {Config.BEST_MODEL_PATH}")

    # =========================================
    # Save results
    # =========================================
    results_txt = os.path.join(Config.OUTPUT_DIR, "results.txt")
    with open(results_txt, "w") as f:
        for r in results_log:
            f.write(
                f"Round {r['round']} - "
                f"Val Loss: {r['val_loss']:.6f}, "
                f"Pixel Accuracy: {r['pixel_accuracy']:.6f}, "
                f"Mean Dice: {r['mean_dice']:.6f}, "
                f"Mean IoU: {r['mean_iou']:.6f}\n"
            )

    results_csv = os.path.join(Config.OUTPUT_DIR, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "val_loss", "pixel_accuracy", "mean_dice", "mean_iou"]
        )
        writer.writeheader()
        writer.writerows(results_log)

    # =========================================
    # Plot results
    # =========================================
    rounds = [r["round"] for r in results_log]
    losses = [r["val_loss"] for r in results_log]
    accuracies = [r["pixel_accuracy"] for r in results_log]
    dice_scores = [r["mean_dice"] for r in results_log]
    iou_scores = [r["mean_iou"] for r in results_log]

    plt.figure()
    plt.plot(rounds, losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Validation Loss")
    plt.title("Federated Training Loss")
    plt.grid()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(rounds, accuracies, marker="o", label="Accuracy")
    plt.plot(rounds, dice_scores, marker="o", label="Dice")
    plt.plot(rounds, iou_scores, marker="o", label="IoU")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Federated Metrics per Round")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "metrics_plot.png"))
    plt.close()

    centralized = {
        "accuracy": 0.975,
        "dice": 0.886,
        "iou": 0.810
    }

    last = results_log[-1]
    labels = ["Accuracy", "Dice", "IoU"]
    fed_values = [last["pixel_accuracy"], last["mean_dice"], last["mean_iou"]]
    cen_values = [centralized["accuracy"], centralized["dice"], centralized["iou"]]
    x = range(len(labels))

    plt.figure()
    plt.bar(x, fed_values, width=0.4, label="Federated")
    plt.bar([i + 0.4 for i in x], cen_values, width=0.4, label="Centralized")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Score")
    plt.title("Federated vs Centralized Comparison")
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "comparison_plot.png"))
    plt.close()

    # =========================================
    # Final notebook-style table
    # =========================================
    print("\nFinal Training and Validation Results")
    print("=" * 75)
    print(f"{'Round':<10}{'Val Loss':<15}{'Accuracy':<15}{'Dice':<15}{'IoU':<15}")
    print("=" * 75)

    for r in results_log:
        print(
            f"{r['round']:<10}"
            f"{r['val_loss']:<15.6f}"
            f"{r['pixel_accuracy']:<15.6f}"
            f"{r['mean_dice']:<15.6f}"
            f"{r['mean_iou']:<15.6f}"
        )

    print("=" * 75)

    print("\nResults saved:")
    print(results_txt)
    print(results_csv)

    print("\n[FILES] Output directory contents:")
    for file_name in os.listdir(Config.OUTPUT_DIR):
        print(" -", file_name)

    print("\nTraining Done!")


if __name__ == "__main__":
    main()