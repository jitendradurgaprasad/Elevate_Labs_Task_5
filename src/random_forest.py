import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_heart.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("RANDOM FOREST CLASSIFIER")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------

y_pred = rf_model.predict(X_test)


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

confusion = confusion_matrix(
    y_test,
    y_pred
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(f"\nNumber of trees: {rf_model.n_estimators}")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion)


# ---------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------

results_path = OUTPUT_DIR / "random_forest_results.txt"

with open(results_path, "w", encoding="utf-8") as file:

    file.write("RANDOM FOREST CLASSIFIER RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Training samples: {X_train.shape[0]}\n")
    file.write(f"Testing samples: {X_test.shape[0]}\n\n")

    file.write("Model configuration:\n")
    file.write(
        "RandomForestClassifier("
        "n_estimators=100, random_state=42)\n\n"
    )

    file.write("Evaluation Metrics:\n")
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1-score : {f1:.4f}\n\n")

    file.write("Classification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Disease"],
            zero_division=0
        )
    )

    file.write("\nConfusion Matrix:\n")
    file.write(str(confusion))


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("\nResults saved to:")
print(results_path)

print("\nRandom Forest training and evaluation completed successfully.")