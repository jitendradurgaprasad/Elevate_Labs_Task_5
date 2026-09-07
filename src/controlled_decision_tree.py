import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
print("CONTROLLED DECISION TREE")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]


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


# ---------------------------------------------------------
# Train controlled Decision Tree
# ---------------------------------------------------------

MAX_DEPTH = 3

dt_model = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    random_state=42
)

dt_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

train_predictions = dt_model.predict(X_train)
test_predictions = dt_model.predict(X_test)


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

accuracy_gap = train_accuracy - test_accuracy

confusion = confusion_matrix(
    y_test,
    test_predictions
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONTROLLED TREE PERFORMANCE")
print("=" * 60)

print(f"\nMaximum depth : {MAX_DEPTH}")
print(f"Actual depth  : {dt_model.get_depth()}")
print(f"Number leaves : {dt_model.get_n_leaves()}")

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy : {test_accuracy:.4f}")
print(f"Accuracy Gap     : {accuracy_gap:.4f}")

print(f"\nPrecision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_path = OUTPUT_DIR / "controlled_decision_tree_results.txt"

with open(results_path, "w", encoding="utf-8") as file:

    file.write("CONTROLLED DECISION TREE RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n\n")

    file.write("Model configuration:\n")
    file.write(
        f"DecisionTreeClassifier(max_depth={MAX_DEPTH}, random_state=42)\n\n"
    )

    file.write(f"Maximum depth: {MAX_DEPTH}\n")
    file.write(f"Actual tree depth: {dt_model.get_depth()}\n")
    file.write(f"Number of leaves: {dt_model.get_n_leaves()}\n\n")

    file.write("Evaluation Metrics:\n")
    file.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    file.write(f"Testing Accuracy : {test_accuracy:.4f}\n")
    file.write(f"Accuracy Gap     : {accuracy_gap:.4f}\n")
    file.write(f"Precision        : {precision:.4f}\n")
    file.write(f"Recall           : {recall:.4f}\n")
    file.write(f"F1-score         : {f1:.4f}\n\n")

    file.write("Classification Report:\n")
    file.write(
        classification_report(
            y_test,
            test_predictions,
            target_names=["No Disease", "Disease"],
            zero_division=0
        )
    )

    file.write("\nConfusion Matrix:\n")
    file.write(str(confusion))

    file.write("\n\nInterpretation:\n")
    file.write(
        "The Decision Tree depth was restricted to 3 based on the "
        "depth analysis. This substantially reduces the training-testing "
        "accuracy gap compared with the unrestricted tree while "
        "maintaining the same observed test accuracy on the selected "
        "train-test split.\n"
    )


# ---------------------------------------------------------
# Visualize controlled tree
# ---------------------------------------------------------

plt.figure(figsize=(20, 12))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=10
)

plt.title(
    "Controlled Decision Tree - Maximum Depth 3",
    fontsize=18,
    pad=20
)

plt.tight_layout()

tree_path = OUTPUT_DIR / "controlled_decision_tree.png"

plt.savefig(
    tree_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("\nResults saved to:")
print(results_path)

print("\nControlled tree visualization saved to:")
print(tree_path)

print("\nControlled Decision Tree completed successfully.")