<p align="center">
  <img src="./assets/physics-banner.svg" alt="Computational Physics Projects" width="100%">
</p>

<p align="center">
  <strong>Machine learning, physical models, and data-driven experiments</strong>
</p>

## Table of contents

| Project | What it does | Open |
|---|---|---|
| **MAGIC Telescope Classification** | Compares classifiers that distinguish gamma-ray events from hadron background using the MAGIC dataset. | [Open `magic_telescope/`](./magic_telescope/) |
| **Pendulum ANN** | Generates synthetic pendulum data and trains a feedforward neural network to predict oscillation period. | [Open `pendulum_ann/`](./pendulum_ann/) |

Each project folder contains its own README with dataset details, methodology,
model-specific commands, and results.

## Getting started

The projects use Python. For MAGIC, create and activate a virtual environment
with Python 3.13.0 from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the tested, pinned dependencies from the root requirements file:

```powershell
python -m pip install -r requirements.txt
```

The pendulum ANN project has its own dependency list. Use a separate virtual
environment and follow its README to avoid changing the tested MAGIC
environment.

## Run a project

Run these commands from the repository root after installing dependencies:

```powershell
# MAGIC gamma-ray classification
python -m magic_telescope.random_forest_classifier

# Pendulum ANN: generate data, then train and evaluate
python -m pendulum_ann.generate_dataset
python -m pendulum_ann.train_model
```

The MAGIC project expects its dataset in its folder. The pendulum project
generates its own reproducible synthetic dataset. See each project's README
for all model commands, optional arguments, outputs, and interpretation of
its results.

## Run the MAGIC tests

```powershell
python -m unittest discover -s magic_telescope -p "test_*.py" -v
```

GitHub Actions also runs these tests on pushes and pull requests.

## Repository layout

```text
magic_telescope/   MAGIC event-classification project and dataset
pendulum_ann/      Pendulum-period ANN, dataset generator, and outputs
```
