import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


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
print("DECISION TREE VISUALIZATION")
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
# Train Decision Tree
# ---------------------------------------------------------

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Display tree information
# ---------------------------------------------------------

print(f"\nTree depth: {dt_model.get_depth()}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")


# ---------------------------------------------------------
# Visualize Decision Tree
# ---------------------------------------------------------

plt.figure(figsize=(30, 18))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=8
)

plt.title(
    "Decision Tree for Heart Disease Prediction",
    fontsize=20,
    pad=20
)

plt.tight_layout()


# ---------------------------------------------------------
# Save visualization
# ---------------------------------------------------------

output_path = OUTPUT_DIR / "decision_tree.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nDecision Tree visualization saved to:")
print(output_path)

print("\nVisualization completed successfully.")