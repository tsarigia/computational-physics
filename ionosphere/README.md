# Ionosphere Radar-Return Classification

This project starts with exploratory data analysis of the UCI Ionosphere
dataset. It examines radar-return labels, feature distributions, correlations,
and univariate relationships with the class before model comparisons.

## Dataset

The dataset contains 351 radar-return observations, 34 numeric attributes,
and a binary class:

| Label | Meaning | Encoding |
|---|---|---:|
| `b` | Bad return: no evidence of ionospheric structure | 0 |
| `g` | Good return: evidence of ionospheric structure | 1 |

The data were collected by a high-frequency radar system in Goose Bay,
Labrador. The attributes are processed measurements derived from complex
signal autocorrelation values for 17 pulse numbers; they are not direct
measurements of electron density. The supplied description does not provide
individual physical names for the 34 columns, so code and plots use neutral
names `Attribute_01` through `Attribute_34`.

The local data contain 225 good and 126 bad observations. `Attribute_02` is
constant at zero in the supplied file, despite the dataset description
stating that all predictors are continuous. The EDA reports this feature but
does not silently remove or transform it. The original measurements are
preserved.

## Setup

From the repository root, create and activate a virtual environment and
install the pinned project dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the exploratory analysis

```powershell
python -m ionosphere.explore
```

By default, the script reads `ionosphere/data/ionosphere.data` and writes four
figures to `ionosphere/outputs/`. Use `--data-path` and `--output-dir` to
choose different paths.

The figures show:

- Class counts and percentages.
- Per-feature histograms split by good/bad class.
- Pearson correlations between non-constant features.
- Point-biserial correlations between each non-constant feature and the binary
  class (`g=1`, `b=0`).

The feature/class correlations are univariate linear associations, not causal
effects or feature-importance scores. The feature-feature heatmap excludes
constant attributes because Pearson correlation is undefined for them.
Exploratory correlations are calculated on the full dataset; they are not
used to claim model performance or to remove features. Any scaling or feature
selection for later models must be fitted inside each cross-validation fold
to avoid leakage.

## Initial fixed-parameter comparison

Run the baseline and initial classifier set from the repository root:

```powershell
python -m ionosphere.model_comparison
```

On Windows systems without `wmic`, set
`$env:LOKY_MAX_CPU_COUNT = "1"` before running the command to avoid joblib's
physical-core detection warning.

The comparison includes a prior-class Dummy baseline, Logistic Regression,
linear and RBF SVM, 5-neighbor KNN, Random Forest, Gradient Boosting, and
XGBoost. SVM, KNN, and Logistic Regression use `StandardScaler` inside a
pipeline, so scaling is fitted using only each training fold. The two
tree-ensemble methods do not require scaling.

All models use the same shuffled, stratified 5-fold cross-validation splits
with `random_state=42` and fixed, documented-in-code parameters; this stage
does not tune models. The table reports mean ± standard deviation across folds.
Precision, recall, and F1 use `g` (encoded as 1) as the positive class.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.641 ± 0.004 | 0.500 ± 0.000 | 0.641 ± 0.004 | 1.000 ± 0.000 | 0.781 ± 0.003 | 0.500 ± 0.000 |
| Logistic Regression | 0.886 ± 0.051 | 0.855 ± 0.060 | 0.872 ± 0.043 | 0.964 ± 0.034 | 0.916 ± 0.037 | 0.896 ± 0.072 |
| Linear SVM | 0.900 ± 0.052 | 0.873 ± 0.064 | 0.888 ± 0.054 | 0.969 ± 0.037 | 0.926 ± 0.038 | 0.883 ± 0.069 |
| RBF SVM | **0.940 ± 0.028** | **0.924 ± 0.034** | 0.929 ± 0.029 | 0.982 ± 0.019 | **0.955 ± 0.020** | **0.981 ± 0.015** |
| K-Nearest Neighbors | 0.840 ± 0.024 | 0.786 ± 0.030 | 0.812 ± 0.021 | 0.978 ± 0.016 | 0.887 ± 0.016 | 0.907 ± 0.050 |
| Random Forest | 0.932 ± 0.021 | 0.919 ± 0.018 | **0.931 ± 0.008** | 0.964 ± 0.034 | 0.947 ± 0.017 | 0.980 ± 0.014 |
| Gradient Boosting | 0.926 ± 0.045 | 0.909 ± 0.049 | 0.920 ± 0.034 | 0.969 ± 0.037 | 0.944 ± 0.034 | 0.965 ± 0.019 |
| XGBoost | 0.934 ± 0.028 | 0.916 ± 0.033 | 0.921 ± 0.026 | 0.982 ± 0.019 | 0.951 ± 0.021 | 0.965 ± 0.020 |

### Preliminary findings

- The RBF SVM has the strongest mean ROC-AUC, balanced accuracy, accuracy,
  and F1 in this initial comparison. Random Forest is close in ROC-AUC
  (0.980 vs. 0.981) and has the highest mean precision.
- All trained models outperform the majority-class baseline on balanced
  accuracy and ROC-AUC. The baseline's 1.000 recall is not evidence of a good
  classifier: it predicts every observation as `g` and has only 0.500
  balanced accuracy.
- KNN is weaker than the other trained models with the current fixed
  5-neighbor setting. This does not rule it out after careful tuning.

These are preliminary cross-validation estimates, not a held-out test result.
The dataset is small (351 observations), and the same folds were used to
compare all models. The nested comparison below tunes model parameters only
inside each outer training fold.

## Nested cross-validation

Run nested cross-validation from the repository root:

```powershell
python -m ionosphere.model_comparison --nested
```

The default design uses 5 stratified outer folds and 5 stratified inner folds.
Each model's hyperparameters are selected in the inner folds using ROC-AUC;
the selected estimator is then scored on the corresponding outer fold. The
script reports outer-fold mean ± standard deviation for the same six metrics,
as well as the best inner-fold parameters selected in each outer fold. The
bounded search grids are defined in `ionosphere/model_comparison.py`.

This estimates the performance of tuning each preselected model family; if
the best family is chosen after comparing outer-fold results, that final
selection can still be optimistic. A final performance claim should be
confirmed with nested model-family selection or a separate untouched test set.

### Nested-CV results

The following results are the mean ± standard deviation across the 5 outer
folds. The inner folds selected hyperparameters using ROC-AUC.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.641 ± 0.004 | 0.500 ± 0.000 | 0.641 ± 0.004 | 1.000 ± 0.000 | 0.781 ± 0.003 | 0.500 ± 0.000 |
| Logistic Regression | 0.869 ± 0.046 | 0.830 ± 0.058 | 0.850 ± 0.043 | 0.969 ± 0.020 | 0.905 ± 0.031 | 0.889 ± 0.091 |
| Linear SVM | 0.875 ± 0.047 | 0.835 ± 0.063 | 0.854 ± 0.048 | 0.973 ± 0.010 | 0.909 ± 0.031 | 0.880 ± 0.051 |
| RBF SVM | **0.946 ± 0.040** | **0.928 ± 0.052** | 0.931 ± 0.048 | 0.991 ± 0.012 | **0.960 ± 0.029** | 0.976 ± 0.016 |
| K-Nearest Neighbors | 0.826 ± 0.030 | 0.767 ± 0.036 | 0.798 ± 0.025 | 0.978 ± 0.016 | 0.878 ± 0.020 | 0.908 ± 0.048 |
| Random Forest | 0.931 ± 0.042 | 0.918 ± 0.051 | **0.932 ± 0.044** | 0.964 ± 0.025 | 0.948 ± 0.032 | **0.978 ± 0.016** |
| Gradient Boosting | 0.929 ± 0.034 | 0.909 ± 0.047 | 0.919 ± 0.046 | 0.978 ± 0.027 | 0.947 ± 0.024 | 0.967 ± 0.024 |
| XGBoost | 0.929 ± 0.035 | 0.909 ± 0.054 | 0.920 ± 0.059 | 0.978 ± 0.022 | 0.947 ± 0.024 | 0.966 ± 0.026 |

### Nested-CV findings

- RBF SVM has the best mean accuracy, balanced accuracy, and F1. Random
  Forest has the highest mean ROC-AUC and precision, but its ROC-AUC is
  effectively tied with RBF SVM at the reported precision (0.978 vs. 0.976).
- Both models substantially outperform the Dummy baseline on balanced
  accuracy and ROC-AUC. The baseline's perfect recall only reflects that it
  predicts every example as the majority class `g`.
- KNN remains the weakest trained model in this bounded search. Its search
  usually selected 9 neighbors with distance weighting; this is not evidence
  that every KNN configuration performs poorly.

The fold-level best parameters vary, as expected from this small dataset.
These nested results tune each model family independently; choosing the
winner by comparing outer-fold scores can still introduce model-selection
bias. Treat them as stronger but still preliminary estimates, not as final
held-out test performance.

## Data validation and tests

The loader checks the file, its 35-column shape, numeric finite features, and
the expected `b`/`g` labels. It encodes classes as `b=0` and `g=1`, without
scaling feature values.

Run the focused tests from the repository root:

```powershell
python -m unittest discover -s ionosphere -p "test_*.py" -v
```

## Attribution and license

Sigillito, V., Wing, S., Hutton, L., & Baker, K. (1989). *Classification of
radar returns from the ionosphere using neural networks*. Johns Hopkins APL
Technical Digest, 10, 262-266.

Dataset record: [Ionosphere, UCI Machine Learning Repository](https://doi.org/10.24432/C5W01B).
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The project uses the supplied data and adds neutral feature names and numeric
class encoding for analysis.
