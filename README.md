# AI & ML Internship — Task 5
## Decision Trees and Random Forests

### Objective

The objective of this task is to understand and implement tree-based machine learning models for classification. The project covers Decision Tree classification, tree visualization, overfitting analysis, Random Forest classification, feature importance, and cross-validation.

---

## Dataset

The project uses a heart disease dataset containing:

- 1,025 original records
- 13 input features
- 1 binary target variable
- No missing values

During preprocessing, exact duplicate rows were identified and removed.

- Original records: 1,025
- Duplicate records removed: 723
- Cleaned records used for modeling: 302

The original dataset is preserved as `data/heart.csv`, while the cleaned dataset is stored as `data/cleaned_heart.csv`.

### Features

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

Target:

- `target` — binary classification target

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz

---

## Project Workflow

Raw Dataset  
↓  
Data Exploration  
↓  
Duplicate Detection and Cleaning  
↓  
Decision Tree Classifier  
↓  
Decision Tree Visualization  
↓  
Tree Depth and Overfitting Analysis  
↓  
Controlled Decision Tree  
↓  
Random Forest Classifier  
↓  
Feature Importance Analysis  
↓  
5-Fold Cross-Validation  
↓  
Final Model Comparison

---

## Project Structure

```text
Elevate-Task-5/
│
├── data/
│   ├── heart.csv
│   └── cleaned_heart.csv
│
├── outputs/
│   ├── dataset_summary.txt
│   ├── target_distribution.png
│   ├── decision_tree_results.txt
│   ├── decision_tree.png
│   ├── tree_depth_analysis.csv
│   ├── depth_vs_accuracy.png
│   ├── overfitting_analysis.txt
│   ├── controlled_decision_tree_results.txt
│   ├── controlled_decision_tree.png
│   ├── random_forest_results.txt
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── feature_importance_analysis.txt
│   ├── cross_validation_results.csv
│   ├── cross_validation_summary.csv
│   ├── cross_validation_report.txt
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── model_comparison_report.txt
│
├── src/
│   ├── data_exploration.py
│   ├── data_preprocessing.py
│   ├── decision_tree.py
│   ├── visualize_decision_tree.py
│   ├── tree_depth_analysis.py
│   ├── controlled_decision_tree.py
│   ├── random_forest.py
│   ├── feature_importance.py
│   ├── cross_validation.py
│   └── model_comparison.py
│
├── .gitignore
├── requirements.txt
└── README.md