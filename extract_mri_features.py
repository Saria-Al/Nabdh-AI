import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from dataset import ACDCH5SliceDataset, split_patients_from_slices
from deeplab_model import build_deeplab


label_map = {
    "NOR": 0,
    "DCM": 1,
    "HCM": 2,
    "MINF": 3,
    "RV": 4
}

label_names = {
    0: "NOR",
    1: "DCM",
    2: "HCM",
    3: "MINF",
    4: "RV"
}


def ensure_single_channel(imgs):
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)
    elif imgs.dim() == 4 and imgs.shape[1] == 3:
        imgs = imgs.mean(dim=1, keepdim=True)
    return imgs


def extract_features(image, mask):
    image = image.astype(np.float32)
    mask = mask.astype(np.uint8)

    region_pixels = image[mask > 0]
    mask_area = np.sum(mask > 0)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0

        circularity = (
            (4 * np.pi * mask_area) / (perimeter ** 2)
            if perimeter != 0 else 0
        )
    else:
        perimeter = 0
        w, h = 0, 0
        aspect_ratio = 0
        circularity = 0

    return {
        "mask_area": mask_area,
        "perimeter": perimeter,
        "circularity": circularity,
        "bbox_width": w,
        "bbox_height": h,
        "aspect_ratio": aspect_ratio,
        "mean_intensity": float(np.mean(region_pixels)) if len(region_pixels) > 0 else 0,
        "std_intensity": float(np.std(region_pixels)) if len(region_pixels) > 0 else 0,
        "min_intensity": float(np.min(region_pixels)) if len(region_pixels) > 0 else 0,
        "max_intensity": float(np.max(region_pixels)) if len(region_pixels) > 0 else 0,
    }


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    _, val_indices = split_patients_from_slices(
        Config.PREPROCESSED_SLICES_DIR,
        Config.VAL_RATIO,
        Config.SEED
    )

    dataset = ACDCH5SliceDataset(
        slices_dir=Config.PREPROCESSED_SLICES_DIR,
        indices=val_indices,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = build_deeplab(
        num_seg_classes=Config.NUM_SEG_CLASSES
    ).to(Config.DEVICE)

    weights = torch.load(
        Config.BEST_MODEL_PATH,
        map_location=Config.DEVICE
    )

    model.load_state_dict(weights, strict=True)
    model.eval()

    output_csv = os.path.join(
        Config.OUTPUT_DIR,
        "mri_extracted_features.csv"
    )

    rows = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs = batch[0].to(Config.DEVICE)

            disease_label_raw = batch[2][0] if len(batch) > 2 else "Unknown"
            patient_id = batch[3][0] if len(batch) > 3 else f"P{idx:04d}"

            if torch.is_tensor(disease_label_raw):
                disease_label_id = int(disease_label_raw.item())
                disease_label = label_names.get(disease_label_id, "Unknown")
            else:
                disease_label = str(disease_label_raw)
                disease_label_id = label_map.get(disease_label, -1)

            imgs = ensure_single_channel(imgs)

            outputs = model(imgs)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            pred_mask = torch.argmax(logits, dim=1)

            image_np = imgs[0, 0].cpu().numpy()
            mask_np = pred_mask[0].cpu().numpy()

            features = extract_features(image_np, mask_np)

            row = {
                "sample_id": idx,
                "patient_id": patient_id,
                "disease_label": disease_label,
                "disease_label_id": disease_label_id,
                **features
            }

            rows.append(row)

    fieldnames = [
        "sample_id",
        "patient_id",
        "disease_label",
        "disease_label_id",
        "mask_area",
        "perimeter",
        "circularity",
        "bbox_width",
        "bbox_height",
        "aspect_ratio",
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted MRI features saved to: {output_csv}")


if __name__ == "__main__":
    main()