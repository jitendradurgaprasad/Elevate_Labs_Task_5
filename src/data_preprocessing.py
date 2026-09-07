import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "heart.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_heart.csv"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("HEART DISEASE DATA PREPROCESSING")
print("=" * 60)


# ---------------------------------------------------------
# Original dataset information
# ---------------------------------------------------------

original_rows = len(df)

print(f"\nOriginal number of rows: {original_rows}")
print(f"Original number of columns: {df.shape[1]}")


# ---------------------------------------------------------
# Check missing values
# ---------------------------------------------------------

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)


# ---------------------------------------------------------
# Detect duplicate rows
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows detected: {duplicate_count}")


# ---------------------------------------------------------
# Remove exact duplicate rows
# ---------------------------------------------------------

df_cleaned = df.drop_duplicates().reset_index(drop=True)

cleaned_rows = len(df_cleaned)

print(f"Rows after removing duplicates: {cleaned_rows}")
print(f"Rows removed: {original_rows - cleaned_rows}")


# ---------------------------------------------------------
# Check missing values after cleaning
# ---------------------------------------------------------

print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum())


# ---------------------------------------------------------
# Target distribution after cleaning
# ---------------------------------------------------------

print("\nTarget distribution after cleaning:")
print(df_cleaned["target"].value_counts().sort_index())

print("\nTarget distribution percentage after cleaning:")
print(
    df_cleaned["target"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


# ---------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------

df_cleaned.to_csv(
    CLEANED_DATA_PATH,
    index=False
)


# ---------------s------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Original rows : {original_rows}")
print(f"Cleaned rows  : {cleaned_rows}")
print(f"Rows removed  : {original_rows - cleaned_rows}")

print(f"\nCleaned dataset saved to:")
print(CLEANED_DATA_PATH)