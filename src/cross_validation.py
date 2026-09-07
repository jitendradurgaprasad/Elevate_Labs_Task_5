import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_heart.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

X = df.drop("target", axis=1)
y = df["target"]


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Define models
models = {
    "Controlled Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# Evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


all_results = []


# Evaluate each model
for model_name, model in models.items():

    print(f"\n{'-' * 60}")
    print(model_name)
    print(f"{'-' * 60}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    for fold in range(5):
        result = {
            "Model": model_name,
            "Fold": fold + 1,
            "Accuracy": scores["test_accuracy"][fold],
            "Precision": scores["test_precision"][fold],
            "Recall": scores["test_recall"][fold],
            "F1": scores["test_f1"][fold]
        }

        all_results.append(result)

        print(
            f"Fold {fold + 1}: "
            f"Accuracy={result['Accuracy']:.4f}, "
            f"Precision={result['Precision']:.4f}, "
            f"Recall={result['Recall']:.4f}, "
            f"F1={result['F1']:.4f}"
        )

    print("\nMean scores:")

    print(f"Accuracy : {scores['test_accuracy'].mean():.4f}")
    print(f"Precision: {scores['test_precision'].mean():.4f}")
    print(f"Recall   : {scores['test_recall'].mean():.4f}")
    print(f"F1-score : {scores['test_f1'].mean():.4f}")

    print("\nStandard deviation:")

    print(f"Accuracy : {scores['test_accuracy'].std():.4f}")
    print(f"Precision: {scores['test_precision'].std():.4f}")
    print(f"Recall   : {scores['test_recall'].std():.4f}")
    print(f"F1-score : {scores['test_f1'].std():.4f}")


# Save fold-level results
results_df = pd.DataFrame(all_results)

csv_path = OUTPUT_DIR / "cross_validation_results.csv"
results_df.to_csv(csv_path, index=False)


# Create summary
summary = results_df.groupby("Model").agg(
    Accuracy_Mean=("Accuracy", "mean"),
    Accuracy_Std=("Accuracy", "std"),
    Precision_Mean=("Precision", "mean"),
    Precision_Std=("Precision", "std"),
    Recall_Mean=("Recall", "mean"),
    Recall_Std=("Recall", "std"),
    F1_Mean=("F1", "mean"),
    F1_Std=("F1", "std")
).reset_index()


summary_path = OUTPUT_DIR / "cross_validation_summary.csv"
summary.to_csv(summary_path, index=False)


# Save readable report
report_path = OUTPUT_DIR / "cross_validation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("5-FOLD CROSS-VALIDATION REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write("Dataset:\n")
    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Features: {X.shape[1]}\n\n")

    file.write("Cross-validation setup:\n")
    file.write("Method: Stratified 5-Fold Cross-Validation\n")
    file.write("Shuffle: True\n")
    file.write("Random State: 42\n\n")

    file.write("Model summary:\n\n")
    file.write(summary.to_string(index=False))

    file.write("\n\nInterpretation:\n")
    file.write(
        "Cross-validation evaluates model performance across multiple "
        "train-validation splits. Mean scores represent average performance "
        "across the five folds, while standard deviation indicates how much "
        "the performance varies between folds. Lower variation generally "
        "indicates more consistent performance.\n"
    )


print("\n" + "=" * 60)
print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFold-level results saved to:")
print(csv_path)

print("\nSummary saved to:")
print(summary_path)

print("\nReport saved to:")
print(report_path)