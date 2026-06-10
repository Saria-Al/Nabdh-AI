import os
import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    DATA_ROOT = "./ACDC_extracted"
    PREPROCESSED_SLICES_DIR = os.path.join(
        DATA_ROOT,
        "Cardic disease complete code and data",
        "ACDC_preprocessed",
        "ACDC_training_slices"
    )

    NUM_SEG_CLASSES = 4
    NUM_DISEASE_CLASSES = 5  

    NUM_CLIENTS = 5
    GLOBAL_ROUNDS = 3
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 8
    LR = 1e-4
    VAL_RATIO = 0.2

    OUTPUT_DIR = "fl_outputs"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "fed_deeplab_best.pt")