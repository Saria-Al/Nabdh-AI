import os
import pandas as pd

GROUP_TO_CLASS = {
    "NOR": 0,
    "DCM": 1,
    "HCM": 2,
    "MINF": 3,
    "RV": 4,
}

def main():
    # عدّلي هذا إذا كان train.csv موجودًا في مكان آخر
    train_csv = "train.csv"

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            "Could not find train.csv in project folder. "
            "Put the train.csv from your original notebook/dataset here."
        )

    df = pd.read_csv(train_csv)

    required = {"pid", "pathology"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")

    df["pid"] = df["pid"].astype(str).str.strip()
    df["pathology"] = df["pathology"].astype(str).str.upper().str.strip()

    df = df[df["pathology"].isin(GROUP_TO_CLASS.keys())].copy()
    df["disease_label"] = df["pathology"].map(GROUP_TO_CLASS)

    out_df = df[["pid", "pathology", "disease_label"]].drop_duplicates().sort_values("pid")
    out_df.to_csv("pathology_map.csv", index=False)

    print("Saved pathology map to pathology_map.csv")
    print(out_df.head())

if __name__ == "__main__":
    main()