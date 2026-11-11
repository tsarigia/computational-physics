# Pendulum Period Regression with an ANN

This standalone project trains a feedforward neural network to predict the
period of a pendulum from its length, initial angle, and gravitational
acceleration. It generates a reproducible synthetic dataset, scales the input
and output values, uses separate training/validation/test sets, and reports
regression errors on the held-out test set.

## Physical model and assumptions

For small angles, the period is approximated by

```text
T0 = 2*pi*sqrt(L/g)
```

For larger angles, this project uses the approximation from the assignment:

```text
T = 2*pi*sqrt(L/g) * (1 + theta0^2/16 + 11*theta0^4/3072)
```

`theta0` is converted from degrees to radians before it is used in the
equation. Each generated period also receives additive Gaussian noise.
Because the assignment does not specify the dataset size or noise level, the
defaults are 10,000 samples and a noise standard deviation of 0.02 seconds.
Both are configurable when generating data.

The generated ranges are:

| Quantity | Range |
|---|---:|
| Pendulum length `L` | 0.2–2.0 m |
| Initial angle `theta0` | 5–80 degrees |
| Gravitational acceleration `g` | 9.6–9.9 m/s² |
| Period noise | Gaussian, mean 0, standard deviation 0.02 s |

The synthetic CSV contains `L_m`, `theta0_rad`, `g_m_s2`, and `period_s`.
The period includes the generated noise.

## Setup

From the repository root, create a virtual environment and install this
project's dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\pendulum_ann\requirements.txt
```

The project has its own `requirements.txt`; the existing projects and
requirements at the repository root are not needed to run it.

## Generate the dataset

```powershell
python .\pendulum_ann\generate_dataset.py
```

This creates `pendulum_ann/data/pendulum_dataset.csv`. To change the number of
samples, noise level, or random seed:

```powershell
python .\pendulum_ann\generate_dataset.py --samples 10000 --noise-std 0.02 --seed 42
```

Use `--output` to choose another CSV path.

## Train and evaluate

```powershell
python .\pendulum_ann\train_model.py
```

The script uses the generated CSV by default and saves the test-set
actual-versus-predicted plot to
`pendulum_ann/outputs/actual_vs_predicted.png`. Use `--data-path` or
`--plot-path` to change those paths.

The data is split reproducibly into 70% training, 15% validation, and 15%
test data. Input features and target periods are scaled to `[-1, 1]`; scalers
are fitted only on the training set. The feedforward network has two hidden
layers with 64 and 32 neurons, respectively. Training proceeds one epoch at a
time, and the model with the lowest validation loss is retained. Training
stops if validation loss does not improve for 50 consecutive epochs, up to a
maximum of 1,000 epochs.

The program reports validation and test metrics:

- **MSE:** mean squared error, in s².
- **RMSE:** root mean squared error, in seconds.
- **MAE:** mean absolute error, in seconds.
- **MAPE:** mean absolute percentage error, displayed as a percentage.

The test set is not used to select the training epoch. The generated data is
synthetic and follows the stated approximation plus simulated noise; results
on it do not establish accuracy on real pendulum measurements.

## Project files

- `data_utils.py`: synthetic data generation, saving, and input validation.
- `generate_dataset.py`: command-line dataset generator.
- `train_model.py`: scaling, train/validation/test split, neural-network
  training, metrics, and test plot.
- `requirements.txt`: dependencies for this standalone project.
