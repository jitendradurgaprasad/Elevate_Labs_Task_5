import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


# Test-set results from the completed model evaluations
test_results = pd.DataFrame({
    "Model": [
        "Controlled Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.8033,
        0.7541
    ],
    "Precision": [
        0.7838,
        0.7647
    ],
    "Recall": [
        0.8788,
        0.7879
    ],
    "F1": [
        0.8286,
        0.7761
    ]
})


# Load cross-validation results
cv_path = OUTPUT_DIR / "cross_validation_summary.csv"

if not cv_path.exists():
    raise FileNotFoundError(
        f"Cross-validation summary not found: {cv_path}"
    )

cv_results = pd.read_csv(cv_path)


# Merge test-set and cross-validation results
comparison = test_results.merge(
    cv_results,
    on="Model",
    how="left"
)


# Save complete comparison
comparison_path = OUTPUT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)


print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print("\nTest-set performance:")
print(test_results.to_string(index=False))

print("\n5-Fold Cross-Validation performance:")
print(cv_results.to_string(index=False))

print("\nComplete comparison:")
print(comparison.to_string(index=False))


# Determine best model using cross-validation F1-score
best_model = comparison.loc[
    comparison["F1_Mean"].idxmax(),
    "Model"
]

best_f1 = comparison.loc[
    comparison["F1_Mean"].idxmax(),
    "F1_Mean"
]


# Create comparison chart
plot_data = test_results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
]

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("Decision Tree vs Random Forest Performance")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)

plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

plot_path = OUTPUT_DIR / "model_comparison.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Create final analysis report
report_path = OUTPUT_DIR / "model_comparison_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("FINAL MODEL COMPARISON REPORT\n")
    file.write("=" * 70 + "\n\n")

    file.write("Models evaluated:\n")
    file.write("1. Controlled Decision Tree (max_depth=3)\n")
    file.write("2. Random Forest (100 estimators)\n\n")

    file.write("Test-set performance:\n\n")
    file.write(test_results.to_string(index=False))

    file.write("\n\n5-Fold Cross-Validation performance:\n\n")
    file.write(cv_results.to_string(index=False))

    file.write("\n\nConclusion:\n")

    file.write(
        f"Based on the mean cross-validation F1-score, "
        f"the better-performing model in this experiment is "
        f"{best_model}, with a mean F1-score of {best_f1:.4f}.\n\n"
    )

    file.write(
        "Model selection should consider both predictive performance "
        "and model complexity. The controlled Decision Tree provides "
        "a simpler and more interpretable model, while the Random "
        "Forest uses an ensemble of decision trees and can provide "
        "more robust predictions in many datasets.\n"
    )


print("\nBest model based on cross-validation F1-score:")
print(f"{best_model} ({best_f1:.4f})")

print("\nComparison CSV saved to:")
print(comparison_path)

print("\nComparison chart saved to:")
print(plot_path)

print("\nComparison report saved to:")
print(report_path)

print("\nModel comparison completed successfully.")