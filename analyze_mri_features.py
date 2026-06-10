import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import Config


# =========================================
# Load CSV
# =========================================

csv_path = os.path.join(
    Config.OUTPUT_DIR,
    "mri_extracted_features.csv"
)

df = pd.read_csv(csv_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst Rows:")
print(df.head())


# =========================================
# Create plots directory
# =========================================

plots_dir = os.path.join(
    Config.OUTPUT_DIR,
    "feature_analysis"
)

os.makedirs(plots_dir, exist_ok=True)


# =========================================
# Mean feature values per disease
# =========================================

numeric_cols = [
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

print("\nMean values per disease:")
print(df.groupby("disease_label")[numeric_cols].mean())


# =========================================
# DCM vs HCM comparison
# =========================================

comparison_df = df[
    df["disease_label"].isin(["DCM", "HCM"])
]

print("\nDCM vs HCM:")
print(
    comparison_df.groupby("disease_label")[numeric_cols].mean()
)


# =========================================
# Boxplots
# =========================================

for feature in numeric_cols:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="disease_label",
        y=feature
    )

    plt.title(f"{feature} by Disease")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            f"boxplot_{feature}.png"
        )
    )

    plt.close()


# =========================================
# Histograms
# =========================================

for feature in numeric_cols:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue="disease_label",
        kde=True,
        bins=20
    )

    plt.title(f"{feature} Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            f"hist_{feature}.png"
        )
    )

    plt.close()


# =========================================
# Correlation Matrix
# =========================================

plt.figure(figsize=(12, 10))

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_dir,
        "correlation_matrix.png"
    )
)

plt.close()


# =========================================
# PCA
# =========================================

X = df[numeric_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "disease_label": df["disease_label"]
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PCA1",
    y="PCA2",
    hue="disease_label"
)

plt.title("PCA Visualization")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_dir,
        "pca_plot.png"
    )
)

plt.close()


# =========================================
# t-SNE
# =========================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "disease_label": df["disease_label"]
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="disease_label"
)

plt.title("t-SNE Visualization")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_dir,
        "tsne_plot.png"
    )
)

plt.close()


print("\nFeature analysis completed.")
print(f"Plots saved to: {plots_dir}")