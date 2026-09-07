import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "heart.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

print("=" * 60)
print("HEART DISEASE DATASET EXPLORATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nTarget distribution percentage:")
print(df["target"].value_counts(normalize=True).sort_index() * 100)


# ---------------------------------------------------------
# Save dataset summary
# ---------------------------------------------------------

summary_path = OUTPUT_DIR / "dataset_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("HEART DISEASE DATASET SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Number of rows: {df.shape[0]}\n")
    file.write(f"Number of columns: {df.shape[1]}\n\n")

    file.write("Columns:\n")
    for column in df.columns:
        file.write(f"- {column}\n")

    file.write("\nData types:\n")
    file.write(df.dtypes.to_string())

    file.write("\n\nMissing values:\n")
    file.write(df.isnull().sum().to_string())

    file.write(f"\n\nDuplicate rows: {df.duplicated().sum()}\n")

    file.write("\n\nDescriptive statistics:\n")
    file.write(df.describe().to_string())

    file.write("\n\nTarget distribution:\n")
    file.write(df["target"].value_counts().sort_index().to_string())

    file.write("\n\nTarget distribution percentage:\n")
    file.write(
        (df["target"].value_counts(normalize=True).sort_index() * 100).to_string()
    )


# ---------------------------------------------------------
# Target distribution visualization
# ---------------------------------------------------------

target_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    ["No Disease (0)", "Disease (1)"],
    target_counts.values
)

plt.xlabel("Target Class")
plt.ylabel("Number of Samples")
plt.title("Heart Disease Target Distribution")

plt.tight_layout()

plot_path = OUTPUT_DIR / "target_distribution.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nDataset exploration completed successfully.")
print(f"Summary saved to: {summary_path}")
print(f"Target distribution plot saved to: {plot_path}")