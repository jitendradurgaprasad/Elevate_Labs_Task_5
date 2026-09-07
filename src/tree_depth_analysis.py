import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_heart.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DECISION TREE DEPTH AND OVERFITTING ANALYSIS")
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
# Evaluate different tree depths
# ---------------------------------------------------------

depth_results = []

for depth in range(1, 16):

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    accuracy_gap = train_accuracy - test_accuracy

    depth_results.append({
        "max_depth": depth,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": accuracy_gap,
        "number_of_leaves": model.get_n_leaves()
    })


# ---------------------------------------------------------
# Create results DataFrame
# ---------------------------------------------------------

results = pd.DataFrame(depth_results)

print("\nDepth analysis results:")
print(results.to_string(index=False))


# ---------------------------------------------------------
# Identify best depth based on test accuracy
# ---------------------------------------------------------

best_row = results.loc[
    results["test_accuracy"].idxmax()
]

best_depth = int(best_row["max_depth"])
best_test_accuracy = best_row["test_accuracy"]

print("\n" + "=" * 60)
print("BEST TREE DEPTH")
print("=" * 60)

print(f"\nBest max_depth: {best_depth}")
print(f"Best test accuracy: {best_test_accuracy:.4f}")


# ---------------------------------------------------------
# Save analysis results
# ---------------------------------------------------------

csv_path = OUTPUT_DIR / "tree_depth_analysis.csv"

results.to_csv(
    csv_path,
    index=False
)


# ---------------------------------------------------------
# Plot training and testing accuracy
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    results["max_depth"],
    results["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results["max_depth"],
    results["test_accuracy"],
    marker="o",
    label="Testing Accuracy"
)

plt.axvline(
    best_depth,
    linestyle="--",
    label=f"Best Depth = {best_depth}"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs Accuracy")

plt.xticks(range(1, 16))
plt.legend()
plt.grid(True)

plt.tight_layout()

plot_path = OUTPUT_DIR / "depth_vs_accuracy.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Save textual analysis
# ---------------------------------------------------------

analysis_path = OUTPUT_DIR / "overfitting_analysis.txt"

with open(analysis_path, "w", encoding="utf-8") as file:

    file.write("DECISION TREE DEPTH AND OVERFITTING ANALYSIS\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n\n")

    file.write("Depth results:\n")
    file.write(results.to_string(index=False))

    file.write("\n\n")
    file.write(f"Best max_depth: {best_depth}\n")
    file.write(f"Best test accuracy: {best_test_accuracy:.4f}\n\n")

    file.write(
        "Interpretation:\n"
        "As tree depth increases, the model becomes more complex. "
        "Training accuracy may continue increasing because the tree "
        "can fit the training data more closely. If testing accuracy "
        "stops improving or decreases while training accuracy remains "
        "high, this indicates overfitting. Controlling max_depth helps "
        "limit model complexity and improve generalization.\n"
    )


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("\nResults saved to:")
print(csv_path)

print("\nAccuracy plot saved to:")
print(plot_path)

print("\nOverfitting analysis saved to:")
print(analysis_path)

print("\nDepth analysis completed successfully.")
