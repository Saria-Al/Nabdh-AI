import os
import random
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo)


def get_disease_label(pid):
    num = int(pid.replace("patient", ""))

    if num <= 20:
        return 0  # NOR
    elif num <= 40:
        return 1  # DCM
    elif num <= 60:
        return 2  # HCM
    elif num <= 80:
        return 3  # MINF
    else:
        return 4  # RV


class ACDCH5SliceDataset(Dataset):
    def __init__(self, slices_dir, indices=None, augment=False):
        self.slices_dir = slices_dir
        self.augment = augment

        self.files = sorted([
            f for f in os.listdir(slices_dir)
            if f.endswith(".h5")
        ])

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        self.samples = []
        for fname in self.files:
            pid = fname.split("_")[0]
            self.samples.append({
                "file_name": fname,
                "patient_id": pid,
                "path": os.path.join(slices_dir, fname)
            })

    def __len__(self):
        return len(self.samples)

    def _basic_aug(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return img, mask

    def __getitem__(self, idx):
        item = self.samples[idx]

        with h5py.File(item["path"], "r") as f:
            img = np.array(f["image"], dtype=np.float32)
            mask = np.array(f["label"], dtype=np.int64)

        if img.ndim == 3:
            z = img.shape[0] // 2
            img = img[z]
            mask = mask[z]

        img = normalize_img(img)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._basic_aug(img, mask)

        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.repeat(3, 1, 1)

        mask = torch.from_numpy(mask).long()
        disease_label = torch.tensor(get_disease_label(item["patient_id"])).long()

        return img, mask, disease_label, item["patient_id"]


def split_patients_from_slices(slices_dir, val_ratio=0.2, seed=42):
    files = sorted([f for f in os.listdir(slices_dir) if f.endswith(".h5")])
    patient_ids = sorted({f.split("_")[0] for f in files})

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    cut = max(1, int(len(patient_ids) * (1 - val_ratio)))
    train_pids = set(patient_ids[:cut])
    val_pids = set(patient_ids[cut:])

    train_indices, val_indices = [], []
    for i, fname in enumerate(files):
        pid = fname.split("_")[0]
        if pid in train_pids:
            train_indices.append(i)
        else:
            val_indices.append(i)

    return train_indices, val_indices


def build_client_subsets_from_indices(slices_dir, train_indices, num_clients=5, seed=42):
    files = sorted([f for f in os.listdir(slices_dir) if f.endswith(".h5")])

    pid_to_indices = defaultdict(list)
    for idx in train_indices:
        pid = files[idx].split("_")[0]
        pid_to_indices[pid].append(idx)

    pids = list(pid_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)

    client_datasets = {}
    for cid in range(num_clients):
        client_pids = pids[cid::num_clients]
        indices = []

        for pid in client_pids:
            indices.extend(pid_to_indices[pid])

        client_datasets[cid] = ACDCH5SliceDataset(
            slices_dir=slices_dir,
            indices=indices,
            augment=True
        )

    return client_datasets