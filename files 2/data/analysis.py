import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load the dataset ──────────────────────────────────────
df = pd.read_csv('data/water_potability.csv')

# ── 2. Basic exploration ─────────────────────────────────────
print("Shape:", df.shape)            # rows × columns
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

# ── 3. Check target column ───────────────────────────────────
print("\nPotability value counts:")
print(df['Potability'].value_counts())
# 0 = Not potable, 1 = Potable

# ── 4. Handle missing values ─────────────────────────────────
df.fillna(df.median(), inplace=True)
print("\nMissing after fill:", df.isnull().sum().sum())

# ── 5. Quick visualization ───────────────────────────────────
plt.figure(figsize=(10, 6))
df.hist(bins=20, figsize=(14, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig('data/distributions.png')
plt.show()

print("\n✅ Dataset loaded and ready!")