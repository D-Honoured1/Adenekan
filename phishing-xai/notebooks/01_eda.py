"""
Notebook 01 — Exploratory Data Analysis
Run as a script or open in Jupyter with: jupytext --to notebook 01_eda.py

Sections
--------
  1. Dataset overview
  2. Class distribution
  3. Feature distributions (numerical)
  4. Correlation heatmap
  5. Feature-by-class box plots (top 10 most discriminative features)
"""

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path().resolve().parent
DATA_PATH = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DROP_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13})

# %% [markdown]
# ## 1. Dataset Overview

# %%
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Shape: {df.shape}")
df.head(3)

# %%
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
X = df.drop(columns=["label"])
y = df["label"]
print(f"Features: {X.shape[1]}  |  Samples: {len(df):,}")
print(f"Missing values: {df.isnull().sum().sum()}")
df.describe().T.round(3)

# %% [markdown]
# ## 2. Class Distribution

# %%
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts()
bars = ax.bar(["Legitimate (0)", "Phishing (1)"], counts.values,
              color=["#2196f3", "#ff4b4b"], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}\n({val/len(y)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig(ROOT / "models" / "eda_class_dist.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Feature Distributions

# %%
# Top 12 numerical features — histogram grid
num_features = X.select_dtypes(include=np.number).columns.tolist()
sample_features = num_features[:12]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, feat in zip(axes.flatten(), sample_features):
    ax.hist(X.loc[y == 0, feat], bins=40, alpha=0.6, color="#2196f3", label="Legitimate", density=True)
    ax.hist(X.loc[y == 1, feat], bins=40, alpha=0.6, color="#ff4b4b", label="Phishing",   density=True)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("")
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions by Class (density)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(ROOT / "models" / "eda_feature_dists.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Correlation Heatmap (top 20 most variable features)

# %%
top20 = X[num_features].std().nlargest(20).index.tolist()
corr = X[top20].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap (top 20 by std)", fontsize=13)
plt.tight_layout()
plt.savefig(ROOT / "models" / "eda_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Box Plots — Most Discriminative Features

# %%
# Use absolute difference of class means as a proxy for discriminative power
means_diff = (
    X[num_features].loc[y == 1].mean() - X[num_features].loc[y == 0].mean()
).abs().nlargest(10)
top10_disc = means_diff.index.tolist()

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for ax, feat in zip(axes.flatten(), top10_disc):
    ax.boxplot(
        [X.loc[y == 0, feat].dropna(), X.loc[y == 1, feat].dropna()],
        labels=["Legit", "Phish"],
        patch_artist=True,
        boxprops=dict(facecolor="#e3f2fd"),
        medianprops=dict(color="#ff4b4b", linewidth=2),
    )
    ax.set_title(feat, fontsize=8)
plt.suptitle("Top 10 Most Discriminative Features (box plots)", fontsize=13)
plt.tight_layout()
plt.savefig(ROOT / "models" / "eda_boxplots.png", bbox_inches="tight")
plt.show()

print("EDA complete. Plots saved to models/")
