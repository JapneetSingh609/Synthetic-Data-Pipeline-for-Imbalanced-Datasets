# Fraud Detection Pipeline

A modular, production-style machine learning workflow for detecting fraudulent transactions in highly imbalanced tabular data. The project is designed to be practical and reproducible: it keeps preprocessing leak-free, applies SMOTE only on training data, evaluates with fraud-focused metrics, and exports benchmark artifacts for downstream analysis or CI/CD pipelines.

## Why this project

Fraud detection datasets are typically extremely imbalanced, so accuracy alone is a misleading performance signal. This pipeline is built around the realities of real-world fraud data:

- class imbalance is preserved during evaluation
- feature engineering is fitted only on training data
- oversampling is applied only to the training split
- multiple models are compared with consistent metrics
- outputs are saved in a headless-friendly format for automation

## What it includes

- leak-free preprocessing for numerical and categorical features
- stratified train/test splitting
- training-only SMOTE resampling
- model benchmarking with tuned and untuned configurations
- fraud-focused metrics such as precision, recall, F1, PR-AUC, and ROC-AUC
- CSV + JSON + plotting exports for reporting and comparison

## Project structure

```text
fraud_detection/
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── preprocessor.py
│       ├── resampler.py
│       ├── models.py
│       └── evaluate.py
├── tests/
│   └── test_pipeline.py
├── main.py
├── transactions.csv
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── README.md
└── output/
    ├── best_hyperparameters.json
    ├── model_comparison_metrics.csv
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── roc_pr_curves.png
```

## Quick start

### Prerequisites

- Python 3.10+
- pip
- optional: virtual environment

### 1. Clone the repo

```bash
git clone <repo-url>
cd fraud_detection
```

### 2. Create and activate a virtual environment

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

### 4. Run the pipeline

```bash
python main.py
```

This will load the default dataset, validate schema, split data, preprocess features, tune models, evaluate performance, and export results to the `output/` directory.

### 5. Run tests

```bash
python -m unittest discover tests
```

## CLI usage

The project exposes a command-line interface in `main.py`.

### Basic run

```bash
python main.py
```

### Skip hyperparameter tuning

```bash
python main.py --no-tune
```

### Use a custom dataset

```bash
python main.py --data-path "path/to/transactions.csv" --output-dir "my_results"
```

### Generate a synthetic sample dataset

```bash
python main.py --generate-sample
```

### Override tuning controls

```bash
python main.py --tune-iter 8 --cv-folds 5
```

### Full CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--data-path` | `transactions.csv` | Input CSV containing transaction data |
| `--target-col` | `is_fraud` | Name of the fraud label column |
| `--test-size` | `0.2` | Fraction used for testing |
| `--random-state` | `42` | Random seed for reproducibility |
| `--output-dir` | `output` | Directory for saved metrics and plots |
| `--save-plots` / `--no-save-plots` | `True` | Save visual evaluation artifacts |
| `--show-plots` | `False` | Display plots interactively |
| `--tune` / `--no-tune` | `True` | Toggle hyperparameter tuning |
| `--tune-iter` | `5` | Number of RandomizedSearchCV iterations |
| `--cv-folds` | `3` | Cross-validation folds used for tuning |
| `--generate-sample` | `False` | Generate synthetic transaction data if needed |

## How the pipeline works

The process is intentionally structured to avoid common ML pitfalls:

1. Load the dataset and validate schema.
2. Split data into training and test sets while preserving class balance.
3. Fit preprocessing transformers only on the training set.
4. Apply SMOTE only to the training data after preprocessing.
5. Train multiple models and tune them with cross-validation.
6. Evaluate on the untouched test set using fraud-relevant metrics.
7. Save metrics, best parameters, and plots to disk.

This makes the workflow safer and more realistic than pipelines that accidentally leak information from validation or test sets into preprocessing or oversampling steps.

## Evaluation strategy

The project prioritizes metrics that matter for fraud detection:

- recall: catches the largest share of fraudulent activity
- precision: reduces false alarms
- F1-score: balances precision and recall
- PR-AUC: useful for heavily imbalanced datasets
- ROC-AUC: provides a ranking view of separation quality

## Benchmark example

Using the included dataset, the pipeline evaluates a fraud rate of roughly 2.21% and compares several models across the same clean test set.

| Model | Recall | Precision | F1-Score | ROC-AUC | PR-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 85.40% | 15.50% | 0.2624 | 0.9394 | 0.5895 | 89.41% |
| Decision Tree | 82.68% | 33.27% | 0.4745 | 0.9382 | 0.6501 | 95.96% |
| Random Forest | 80.64% | 58.70% | 0.6794 | 0.9684 | 0.7870 | 98.32% |

The Random Forest configuration performs best in this benchmark, offering a strong tradeoff between fraud recall and false-positive control.

## Output artifacts

When the pipeline runs, it writes artifacts to the configured output directory (default: `output/`):

- `best_hyperparameters.json`: best parameters selected during tuning
- `model_comparison_metrics.csv`: consolidated model comparison table
- `confusion_matrices.png`: confusion matrices for each model
- `metrics_comparison.png`: comparison chart across key metrics
- `roc_pr_curves.png`: ROC and precision-recall curves

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


