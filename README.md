
<p align="center">
    <a href="https://github.com/gaozhihong73/DynaGP/releases">
    	<img src="https://img.shields.io/github/v/release/gaozhihong73/DynaGP?style=flat-plastic&color=007acc&include_prereleases" alt="Release">
	</a>
    <a href="https://github.com/gaozhihong73/DynaGP/blob/main/LICENSE">
    	<img src="https://img.shields.io/github/license/gaozhihong73/DynaGP?style=flat-plastic&color=42b883" alt="License">
  	</a> 
    <a href="https://github.com/gaozhihong73/DynaGP/commits/main">
    	<img src="https://img.shields.io/github/last-commit/gaozhihong73/DynaGP?style=flat-plastic&color=fb8c00" alt="Last Commit">
  	</a>
    <img src="https://img.shields.io/github/languages/top/gaozhihong73/DynaGP?style=flat-plastic&color=f1e05a" alt="languagges">
  	<a href="https://github.com/gaozhihong73/DynaGP/stargazers">
    	<img src="https://img.shields.io/github/stars/gaozhihong73/DynaGP?style=flat-plastic&color=dfb317" alt="Stars">
  	</a>
</p>



# DynaGS

## Installation

```bash
# (Optional) On AutoDL, enable the network proxy first
source /etc/network_turbo

# Clone the repository
git clone https://github.com/gaozhihong73/DynaGP.git
cd DynaGS

# Install uv if you haven't already (see https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin Python 3.10 in the project
uv python pin 3.10

# Create a virtualenv and install all dependencies (PyTorch CUDA 12.4 wheels
# are pulled automatically via the pytorch-cu124 index defined in pyproject.toml)
uv sync

# Or, to install into the currently-active environment directly:
# uv pip install -e .
```

> **Note:** the legacy `requirements.txt` in the repo is out of date; the
> authoritative dependency list is `pyproject.toml`. In particular, PyTorch
> is pinned to `2.5.1` built for CUDA 12.4, and `pydantic>=2.13` is required.

## Requirement

Core runtime dependencies (declared in `pyproject.toml`):

```
python >= 3.10, < 3.11
torch == 2.5.1       # CUDA 12.4 build via the pytorch-cu124 index
numpy == 1.23.5
scipy == 1.10.1
pandas == 2.0.3
matplotlib == 3.8.2
scikit-learn == 1.3.2
seaborn == 0.13.2
pydantic == 2.13.4
```

Optional dependencies used by the baseline/ConvAE scripts (install manually
when needed):

```
transformers
opencv-python-headless
pyarrow
torchinfo
tqdm
cupy-cuda11x   # for ConvAE
xgboost        # for Baseline_Train.py
scikit-allel   # for Baseline_Train.py
```

## Options and usage

### Training

#### Usage

```bash
# Default Training
python train.py

# Custom Training
python train.py \
  --exp_name maize_single_train \
  --snp_file dataset/maize/snp/snp_50000.npy \
  --pheno_dir dataset/maize/cleaned_pheno \
  --pheno_names GreenCoverage \
  --time_coordinates 10 12 14 15 17 19 24 26 37 44 51 65 80 101 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0001 \
  --dropout 0.3 \
  --degree 3 \
  --ablation_mode none \
  --use_warmup true \
  --use_kfold false
```

#### Optional Parameters

**Experiment & data**

| **Parameter**        | **Description**                                              | **Default Value**                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `--exp_name`         | Experiment name to distinguish different runs                | timestamp                                        |
| `--snp_file`         | Path to the SNP genotype data file (.npy format)             | dataset/maize/snp/snp_50000.npy                  |
| `--pheno_dir`        | Directory containing the phenotype data files                | dataset/maize/cleaned_pheno                      |
| `--test_ratio`       | Ratio of the dataset to be used for testing                  | 0.1                                              |
| `--val_ratio`        | Ratio of the dataset to be used for validation               | 0.1                                              |
| `--pheno_names`      | List of phenotype names to predict (must match filenames)    | GreenCoverage                                    |
| `--time_coordinates` | Real-world time coordinates for phenotype sampling           | 10 12 14 15 17 19 24 <br />26 37 44 51 65 80 101 |
| `--norm_method`      | Strategy for phenotype normalization (global/timepoint/residual_global) | global                                           |
| `--scaler_type`      | Type of scaler function (zscore/minmax/robust)               | zscore                                           |

**Training hyperparameters**

| **Parameter**        | **Description**                                              | **Default Value** |
| -------------------- | ------------------------------------------------------------ | ----------------- |
| `--epochs`           | Maximum number of training iterations                        | 100               |
| `--batch_size`       | Number of samples per training batch (must be a multiple of 2) | 32                |
| `--lr`               | Initial learning rate                                        | 0.0001            |
| `--min_lr`           | Lower bound of the learning rate (cosine annealing endpoint) | 1e-7              |
| `--weight_decay`     | Weight decay coefficient (L2 regularization)                 | 1e-4              |
| `--dropout`          | Dropout rate for neural network layers                       | 0.3               |
| `--grad_clip_norm`   | Max norm for gradient clipping, 0 disables clipping          | 1.0               |
| `--patience`         | Patience for early stopping (epochs without improvement)     | 10                |
| `--seed`             | Global random seed for reproducibility                       | 73                |
| `--is_save`          | Whether to save the trained model and history locally (true/false) | true              |
| `--num_workers`      | Number of DataLoader worker threads                          | 4                 |
| `--use_warmup`       | Whether to enable the learning rate warmup strategy (true/false) | true              |
| `--warmup_epochs`    | Number of epochs for the warmup phase                        | 5                 |
| `--warmup_start_factor` | Warmup start LR multiplier (`base_lr × start_factor`)     | 0.1               |
| `--warmup_end_factor`   | Warmup end LR multiplier (`base_lr × end_factor`)         | 1.0               |

**Model architecture**

| **Parameter**        | **Description**                                              | **Default Value** |
| -------------------- | ------------------------------------------------------------ | ----------------- |
| `--d_model`          | Hidden dimension size of the Transformer                     | 64                |
| `--num_heads`        | Number of heads in the multi-head attention mechanism        | 4                 |
| `--num_layers`       | Number of encoder layers (network depth)                     | 2                 |
| `--d_ff`             | Dimension of the feed-forward network (FFN) hidden layer     | 256               |
| `--num_embeddings`   | Number of SNP categorical embedding classes                  | 4                 |
| `--degree`           | Order of the Legendre polynomial basis (wide path); 0 = additive-only (constant term only) | 3                 |
| `--ablation_mode`    | Ablation mode: none / wide / deep / add                      | none              |

**Validation**

| **Parameter**        | **Description**                                              | **Default Value** |
| -------------------- | ------------------------------------------------------------ | ----------------- |
| `--use_kfold`        | Whether to enable K-fold cross-validation (true/false)       | false             |
| `--n_folds`          | Number of folds for K-fold cross-validation                  | 10                |

> **Note:** `--is_save`, `--use_warmup`, and `--use_kfold` are parsed with a
> `str→bool` helper (`"true"`/`"false"`), so `--use_kfold false` correctly
| sets it to `False` (the old `type=bool` footgun has been fixed).

### Prediction

#### Usage

```bash
# Default Inference
python infer.py

# Custom Inference
python infer.py \
  --model_path outputs/tfpp/maize_data/models/best_model.pt \
  --scaler_path outputs/tfpp/maize_data/scalers/scaler.pkl \
  --output outputs/infer/inference_maize.csv \
  --snp_file dataset/maize/snp/snp_50000.npy \
  --pheno_dir dataset/maize/cleaned_pheno \
  --time_coordinates 10 12 14 15 17 19 24 26 37 44 51 65 80 101 \
  --degree 3 \
  --ablation_mode none
```

#### Optional Parameters

**Paths & prediction mode**

| **Parameter**        | **Description**                                              | **Default Value**                                   |
| -------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| `--model_path`       | Path to the trained model file (.pt)                        | outputs/tfpp/maize_data/<br />models/best_model.pt |
| `--scaler_path`      | Path to the phenotype scaler file (.pkl)                     | outputs/tfpp/maize_data/<br />scalers/scaler.pkl    |
| `--input_snp`        | Path to external SNP data (.npy) for prediction; if omitted, uses the reproduced test set | None                                                |
| `--output`           | Path to save the prediction results in CSV format            | outputs/infer/<br />inference_{timestamp}.csv       |

**Data splitting (only used when `--input_snp` is not provided; must match training)**

| **Parameter**        | **Description**                                              | **Default Value**                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `--snp_file`         | Path to the original SNP data used during training           | dataset/maize/snp/snp_50000.npy                  |
| `--pheno_dir`        | Directory containing the original phenotype data             | dataset/maize/cleaned_pheno                      |
| `--pheno_names`      | Optional explicit phenotype list (overrides the scaler's list; use only if identical to training) | None (use scaler's list)                         |
| `--test_ratio`       | Ratio of the dataset used for testing                        | 0.1                                              |
| `--val_ratio`        | Ratio of the dataset used for validation                     | 0.1                                              |
| `--seed`             | Random seed for dataset splitting                            | 73                                               |
| `--norm_method`      | Normalization strategy used during training                  | global                                           |
| `--scaler_type`      | Scaler function used during training                         | zscore                                           |

**Runtime**

| **Parameter**        | **Description**                                              | **Default Value**                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `--batch_size`       | Number of samples per inference batch                        | 32                                               |
| `--num_workers`      | Number of DataLoader worker threads                          | 4                                                |
| `--dropout`          | Dropout rate at inference time (usually 0.0)                 | 0.0                                              |

**Model architecture (must match training)**

| **Parameter**        | **Description**                                              | **Default Value**                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `--time_coordinates` | Real-world time coordinates for phenotype sampling           | 10 12 14 15 17 19 24 <br />26 37 44 51 65 80 101 |
| `--d_model`          | Hidden dimension size of the Transformer                     | 64                                               |
| `--num_heads`        | Number of heads in the multi-head attention mechanism        | 4                                                |
| `--num_layers`       | Number of encoder layers                                     | 2                                                |
| `--d_ff`             | Dimension of the feed-forward network (FFN) hidden layer     | 256                                              |
| `--num_embeddings`   | Number of SNP categorical embedding classes                  | 4                                                |
| `--degree`           | Order of the Legendre polynomial basis (wide path)           | 3                                                |
| `--ablation_mode`    | Ablation mode: none / wide / deep / add                      | none                                             |
