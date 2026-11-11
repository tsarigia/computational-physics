# MAGIC Gamma Telescope Classification

This project compares machine-learning classifiers on the MAGIC Gamma Telescope
dataset. The task is to distinguish gamma-ray signal events from hadron
background events using ten measurements derived from Cherenkov telescope
shower images.


## Dataset

The project expects `magic04.data` and its description, `magic04.names`, in
this folder. The data file contains 19,020 rows with ten numeric features and
one class label:

| Label in file | Meaning | Label used by the code |
|---|---|---:|
| `g` | Gamma-ray signal | 1 |
| `h` | Hadron background | 0 |

The ten features are `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`,
`fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, and `fDist`. They describe the
shape, size, and orientation of the shower image. The definitions and units
are documented in [`magic04.names`](./magic04.names).

The dataset documentation reports 12,332 gamma events and 6,688 hadron events.
It also warns that hadrons are underrepresented relative to real observations,
and that classifying background as signal is particularly costly. Therefore,
accuracy alone is not an adequate measure of scientific usefulness.

## Setup

From the repository root, create and activate a virtual environment and
install the project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the classifiers

Run commands from the repository root:

```powershell
python -m magic_telescope.tree_classifier
python -m magic_telescope.random_forest_classifier
python -m magic_telescope.gradient_boosting_classifier
python -m magic_telescope.svm_classifier
```

Each script accepts an optional `--data-path` argument. For example:

```powershell
python -m magic_telescope.random_forest_classifier --data-path .\magic_telescope\magic04.data
```

The decision-tree script also saves a visualization to
`magic_telescope/outputs/decision_tree.png` and accepts `--plot-path` to
change its destination.

## Data preparation and evaluation

[`data_utils.py`](./data_utils.py) contains the shared data-loading, splitting,
and metric code. It:

1. Reads the headerless CSV using the ten feature names above and `class` as
   the target column.
2. Rejects missing files, empty datasets, unknown class labels, non-numeric
   features, and non-finite feature values.
3. Encodes `g` as 1 and `h` as 0.
4. Creates a reproducible 70/30 train/test split with stratification and
   `random_state=42`.

All models use the same split. The test set contains 5,706 events: 3,700
gamma events and 2,006 hadron events. Model fitting and hyperparameter search
use the training set; the reported metrics are computed on the held-out test
set.

The decision tree and Gradient Boosting scripts use five-fold cross-validation
on the training set, selecting parameters by ROC-AUC. The Random Forest is a
100-tree baseline and is not tuned. The SVM script compares linear and RBF
kernels at `C=0.1`, `1`, and `10`; it scales features inside a pipeline so
the scaler is fitted on training data only. The six SVM variants are
exploratory comparisons, not cross-validated hyperparameter selection.

The scripts report:

- **Accuracy:** fraction of all test events classified correctly.
- **Gamma precision:** among events predicted as gamma, the fraction that
  really are gamma.
- **Gamma recall:** among actual gamma events, the fraction identified.
- **ROC-AUC:** ranking ability across classification thresholds.
- **Confusion matrix:** rows are actual classes and columns are predicted
  classes, in the order hadron then gamma:

  ```text
  [[actual hadron predicted hadron, actual hadron predicted gamma],
   [actual gamma  predicted hadron, actual gamma  predicted gamma]]
  ```

The `actual hadron predicted gamma` entry is especially important: it counts
background events mistakenly accepted as gamma. Its rate in the last column
below is calculated among the 2,006 hadrons in this particular test set.

## Results

The following are the results from one run with the settings above. Percentages
are rounded to two decimal places. They describe this fixed test split and
should not be interpreted as guaranteed performance on new observations.

| Model / settings | Accuracy | Gamma precision | Gamma recall | ROC-AUC | Hadrons predicted as gamma |
|---|---:|---:|---:|---:|---:|
| Decision Tree (`max_depth=None`, `min_samples_leaf=50`) | 84.56% | 85.19% | 92.22% | 0.8917 | 593 / 2,006 (29.56%) |
| Random Forest (`n_estimators=100`) | **88.33%** | 88.62% | 94.08% | **0.9351** | **447 / 2,006 (22.28%)** |
| Gradient Boosting (`n_estimators=200`, `learning_rate=0.1`, `max_depth=3`) | 88.00% | 87.62% | 94.89% | 0.9309 | 496 / 2,006 (24.73%) |
| SVM, linear (`C=0.1`) | 77.95% | 79.37% | 89.19% | 0.8231 | 858 / 2,006 (42.77%) |
| SVM, linear (`C=1`) | 77.97% | 79.37% | 89.22% | 0.8229 | 858 / 2,006 (42.77%) |
| SVM, linear (`C=10`) | 77.95% | 79.37% | 89.19% | 0.8229 | 858 / 2,006 (42.77%) |
| SVM, RBF (`C=0.1`) | 85.19% | 83.60% | **96.00%** | 0.9076 | 697 / 2,006 (34.75%) |
| SVM, RBF (`C=1`) | 86.65% | 85.38% | 95.81% | 0.9194 | 607 / 2,006 (30.26%) |
| SVM, RBF (`C=10`) | 87.47% | 86.13% | 96.16% | 0.9284 | 573 / 2,006 (28.56%) |

### Reading the results

On this split, the Random Forest has the highest ROC-AUC and accuracy among
the listed runs, and the lowest hadron-to-gamma error count. The RBF SVM
variants identify a larger fraction of gamma events (higher gamma recall),
but also misclassify more hadrons as gamma. This is a trade-off, not a single
universal ranking: the preferred model depends on the acceptable background
rate and the cost of each type of error.

The dataset description recommends comparing classifiers using the ROC curve
at low hadron acceptance rates (including 1%, 2%, 5%, 10%, and 20%). This
project currently reports overall ROC-AUC and confusion matrices at each
model's default decision threshold; it does **not** yet calculate performance
at those specified operating points. In particular, a high ROC-AUC or gamma
recall alone does not demonstrate that a model meets a required background
rejection level.

## Limitations

- The supplied events are Monte Carlo generated, not a representative sample
  of real telescope observations.
- The hadron frequency in the file is known to be lower than in real data.
  Consequently, test-set precision and accuracy may not reflect real-world
  values.
- Results come from one train/test split. The model comparisons, especially
  the several SVM settings inspected on the same test set, are exploratory.
  Selecting a final model based on these results can make its reported test
  performance optimistic.
- No decision threshold has been selected for a target hadron acceptance
  rate, and no independent real-data validation has been performed.

These classifiers are suitable for learning and comparing methods on this
dataset. Further cross-validation, operating-point analysis, and independent
validation on representative real observations would be needed before drawing
scientific or operational conclusions.

## Dataset reference

See [`magic04.names`](./magic04.names) for the dataset description, class
distribution warning, and prior-use references. The accompanying description
attributes the data to Monte Carlo simulations with CORSIKA and cites:

R. K. Bock et al. (2004), “Methods for multidimensional event classification:
a case study using images from a Cherenkov gamma-ray telescope,”
*Nuclear Instruments and Methods in Physics Research A*, 516, 511–528.
