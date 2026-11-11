# Path-Loss Prediction

This project compares regression models for predicting path loss (in dB) from
the `paper.xlsx` dataset. It includes a Decision Tree depth experiment and two
additional baseline models: Random Forest and Gradient Boosting.

## Dataset

The `paper.xlsx` file is stored in this `path_loss/` directory. It contains
18,898 rows of numeric data without a header row. The first four columns are
used as model features and the last column is the target:

| Dataset column | Name in code | Use |
|---|---|---|
| 1 | `Feature_1` | Input feature |
| 2 | `Feature_2` | Input feature |
| 3 | `Feature_3` | Input feature |
| 4 | `Feature_4` | Input feature |
| 5 | `Path_Loss` | Target to predict, in dB |

The source file does not describe the first four columns, so the code uses
neutral names rather than assigning them unverified physical meanings. The
loader checks that the file exists, has five columns, and contains only
numeric, finite values.

## Setup

Run these commands from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pandas uses `openpyxl` to read the Excel file.

## Run the experiments

```powershell
python -m path_loss.decision_tree_regressor
python -m path_loss.random_forest_regressor
python -m path_loss.gradient_boosting_regressor
```

By default, the scripts read `path_loss/paper.xlsx`. To use another Excel
file, pass its path with `--data-path`:

```powershell
python -m path_loss.decision_tree_regressor --data-path .\path_loss\paper.xlsx
```

## Methodology and model settings

All scripts use the same four input features and reproducible data split: 80%
for training and 20% for testing, with `random_state=42`. Each model is fit
only on the training set; the metrics below are calculated on the held-out
test set.

- **Decision Tree:** compares `max_depth` values of 2, 5, 10, 15, 20, 25, and
  `None`. `None` allows the tree to grow without a maximum depth. Other tree
  settings use scikit-learn defaults.
- **Random Forest:** 100 trees, with `random_state=42`.
- **Gradient Boosting:** 200 boosting stages, `learning_rate=0.1`, and
  `max_depth=3` for the individual trees.

## Evaluation metrics

- **MAE (Mean Absolute Error):** the average absolute difference between
  predicted and actual values. It is measured in dB and directly describes
  the average error magnitude.
- **RMSE (Root Mean Squared Error):** the square root of the mean squared
  error, also measured in dB. It penalizes larger errors more than MAE.
- **MAPE (Mean Absolute Percentage Error):** the mean absolute percentage
  error, displayed as a percentage. The target values in this dataset are
  negative but nonzero; MAPE uses the absolute target magnitude in its
  denominator. Lower values indicate better performance on this test set for
  all three metrics.

## Results

The results below were produced by running the scripts on the supplied
dataset, using the settings above and the fixed split with `random_state=42`.
Values are rounded for presentation.

### Decision Tree by maximum depth

| `max_depth` | MAE (dB) | RMSE (dB) | MAPE |
|---:|---:|---:|---:|
| 2 | 7.6331 | 10.8560 | 6.67% |
| 5 | 6.3888 | 9.0259 | 5.49% |
| 10 | 4.3199 | 6.3584 | 3.59% |
| 15 | 2.6869 | 4.4128 | 2.19% |
| 20 | 2.1284 | 3.7305 | 1.72% |
| 25 | 1.9806 | 3.5532 | 1.59% |
| `None` | **1.9718** | **3.5319** | **1.58%** |

### Model comparison

| Model / settings | MAE (dB) | RMSE (dB) | MAPE |
|---|---:|---:|---:|
| Decision Tree (`max_depth=15`) | 2.6869 | 4.4128 | 2.19% |
| Decision Tree (`max_depth=None`) | **1.9718** | 3.5319 | **1.58%** |
| Random Forest (`n_estimators=100`) | 2.2914 | **3.5405** | 1.92% |
| Gradient Boosting (`200`, `0.1`, `depth=3`) | 5.4726 | 7.9267 | 4.73% |

On this particular split, the Decision Tree with no maximum depth has the
lowest MAE and MAPE among the tested models. The Random Forest has an RMSE
nearly identical to the unrestricted tree and substantially lower than the
Gradient Boosting model with the settings used here. Increasing the Decision
Tree depth improves the test metrics in this experiment, with only a small
additional improvement from depth 25 to `None`.

These results do not establish that the unrestricted tree will generalize
better to new data: a deeper tree may overfit. Comparing multiple depths on
the same test set makes this useful as an exploratory experiment, but not as
an unbiased way to select the final model. For a more reliable model choice,
select the depth using cross-validation on the training data only, then use
the test set once for final evaluation.

## Source files

- `data_utils.py`: Excel loading and validation, train/test splitting, and
  metric calculation.
- `decision_tree_regressor.py`: comparison of `max_depth` values.
- `random_forest_regressor.py`: Random Forest training and evaluation.
- `gradient_boosting_regressor.py`: Gradient Boosting training and evaluation.
