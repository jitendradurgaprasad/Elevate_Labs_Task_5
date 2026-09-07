import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
print("RANDOM FOREST FEATURE IMPORTANCE")
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
# Train Random Forest
# ---------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Calculate feature importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# Display feature importance
# ---------------------------------------------------------

print("\nFeature importance ranking:")
print(feature_importance.to_string(index=False))


# ---------------------------------------------------------
# Save feature importance data
# ---------------------------------------------------------

csv_path = OUTPUT_DIR / "feature_importance.csv"

feature_importance.to_csv(
    csv_path,
    index=False
)


# ---------------------------------------------------------
# Visualize feature importance
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()


plot_path = OUTPUT_DIR / "feature_importance.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Save interpretation
# ---------------------------------------------------------

top_feature = feature_importance.iloc[0]

interpretation_path = OUTPUT_DIR / "feature_importance_analysis.txt"

with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS\n")
    file.write("=" * 55 + "\n\n")

    file.write("Feature importance ranking:\n\n")
    file.write(feature_importance.to_string(index=False))

    file.write("\n\n")
    file.write(
        f"Most important feature: {top_feature['Feature']}\n"
    )

    file.write(
        f"Importance score: {top_feature['Importance']:.6f}\n\n"
    )

    file.write(
        "Interpretation:\n"
        "Feature importance indicates the relative contribution "
        "of each feature to the Random Forest's predictive decisions. "
        "A higher importance score indicates that the feature "
        "contributed more to the model's decision-making process "
        "within this trained model. Feature importance should not "
        "be interpreted as proof of causal relationships.\n"
    )


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("\nFeature importance data saved to:")
print(csv_path)

print("\nFeature importance plot saved to:")
print(plot_path)

print("\nFeature importance analysis saved to:")
print(interpretation_path)

print("\nFeature importance analysis completed successfully.")

