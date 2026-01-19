# MIR: Model Inferred Replica

**MIR (Model Inferred Replica)** is a component of the **MIGATE** framework (Model Inferred Replica for Adversarial Generalized Evasion), focusing on the model extraction phase for malware classifiers and antivirus systems.

## Overview

MIR provides a comprehensive framework for extracting surrogate models from black-box malware classifiers and antivirus systems using active learning techniques. The extracted models can achieve high agreement with target models while maintaining low false positive rates.

### Key Features

- **Multiple Target Support**: Extract models from various targets including:
  - EMBER-based classifiers
  - SOREL-20M models (FCNN, LightGBM)
  - Commercial antivirus systems (AV1-AV4)
  
- **Active Learning Strategies**: Multiple sampling methods for efficient query selection:
  - Entropy-based sampling
  - Random sampling
  - Medoids clustering
  - MC-Dropout uncertainty
  - K-center greedy
  - Ensemble-based methods

- **Surrogate Model Types**: Support for various surrogate architectures:
  - Deep Neural Networks (DNN, dualDNN)
  - LightGBM
  - Support Vector Machines (SVM)

- **Feature Space Independence**: Handles scenarios where target and attacker use different feature spaces through automatic feature alignment

## Setup

### Prerequisites

- Python 3.8+
- Virtual environment support

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:hytong05/model-stealing.git
   cd model-stealing
   ```

2. **Create and activate virtual environment**:
   ```bash
   ./setup_venv.sh
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The `ember` package needs to be installed separately:
   ```bash
   pip install git+https://github.com/endgameinc/ember.git
   ```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

## Project Structure

```
model-stealing/
├── src/                           # Core library
│   ├── attackers/                 # Attack implementations
│   ├── targets/                   # Target model interfaces
│   ├── sampling/                  # Active learning strategies
│   ├── models/                    # Surrogate model architectures
│   ├── datasets/                  # Dataset loaders
│   └── utils/                     # Utility functions
├── scripts/                       # Executable utilities
│   ├── attacks/                   # Extraction pipelines
│   │   ├── model_extraction.py    # Main extraction script
│   │   ├── extract_final_model.py
│   │   └── evaluate_surrogate_similarity.py
│   ├── oracle/                    # Black-box query interfaces
│   ├── data/                      # Data preprocessing
│   ├── inference/                 # Model inference utilities
│   └── examples/                  # Example usage scripts
├── config/                        # Configuration files
├── data/                          # Datasets (not tracked)
├── artifacts/                     # Model artifacts (not tracked)
│   └── targets/                   # Target model files
├── output/                        # Experiment outputs (not tracked)
├── storage/                       # Training checkpoints (not tracked)
├── logs/                          # Log files (not tracked)
├── docs/                          # Documentation
│   ├── SETUP.md                   # Detailed setup guide
│   ├── ORACLE_USAGE.md            # Oracle query documentation
│   └── reports/                   # Generated reports
├── notebooks/                     # Jupyter notebooks
├── examples/                      # Example scripts
├── requirements.txt               # Python dependencies
└── setup_venv.sh                  # Virtual environment setup script
```

## Usage

### Basic Model Extraction

The main extraction script is `scripts/attacks/model_extraction.py`. It supports various parameters for configuring the extraction attack:

```bash
python scripts/attacks/model_extraction.py \
  --data_dir /path/to/data \
  --dataset {ember,sorel,AV} \
  --target_model {ember,sorel-FCNN,sorel-LGB,AV1,AV2,AV3,AV4} \
  --type {DNN,dualDNN,LGB,SVM} \
  --method {entropy,random,medoids,mc_dropout,k-center,ensemble} \
  --budget 2500 \
  --num_queries 10 \
  --num_epochs 1 \
  --log_dir /path/to/logs \
  --fpr 0.006 \
  --seed 42
```

### Parameters

- `--data_dir`: Directory containing the dataset
- `--dataset`: Dataset type (`ember`, `sorel`, or `AV`)
- `--target_model`: Target model to extract from
- `--type`: Surrogate model architecture
- `--method`: Active learning sampling strategy
- `--budget`: Total query budget
- `--num_queries`: Number of query rounds
- `--num_epochs`: Training epochs per round
- `--log_dir`: Output directory for logs and models
- `--fpr`: False positive rate threshold for metrics
- `--seed`: Random seed for reproducibility

### Example: Extracting from SOREL-FCNN

```bash
python scripts/attacks/model_extraction.py \
  --data_dir /data/sorel-data \
  --dataset sorel \
  --target_model sorel-FCNN \
  --type LGB \
  --method medoids \
  --budget 2500 \
  --num_queries 10 \
  --num_epochs 1 \
  --log_dir /tmp/logs/ \
  --fpr 0.006 \
  --seed 42
```

This command will:
1. Extract a LightGBM surrogate model from the SOREL-FCNN target
2. Use medoids clustering for query selection
3. Perform 10 query rounds with a total budget of 2500 queries
4. Save results to `/tmp/logs/`

## Oracle Query Interface

The framework provides a modular oracle interface for querying target models. This ensures complete separation between the attack process and target model access.

### Local Oracle (No Server Required)

For local model files, use the `scripts/oracle/query_labels.py` script:

**For Keras/TensorFlow models (H5)**:
```bash
python scripts/oracle/query_labels.py \
  --input-path data/pool_features.npy \
  --output-path cache/pool_labels.npy \
  --model-type h5 \
  --model-path artifacts/targets/target_model.h5
```

**For LightGBM models**:
```bash
python scripts/oracle/query_labels.py \
  --input-path data/pool.parquet \
  --output-path cache/pool_labels.npy \
  --model-type lgb \
  --model-path artifacts/targets/target_model.lgb \
  --normalization-stats-path artifacts/targets/target_normalization_stats.npz
```

The extraction scripts can use pre-generated labels or directly import the oracle client for on-the-fly queries.

## Feature Space Independence

MIR handles scenarios where target and attacker models use different feature spaces:

- **For Antivirus Systems**: Uses raw binary files as the common interface. The AV processes files with its internal features, while the surrogate learns on attacker-defined features (e.g., EMBER features).

- **For ML Models**: Automatically performs feature alignment. If the attacker's feature space is larger than the target's, excess features are trimmed. The framework ensures interface compliance while maintaining feature space independence.

## Evaluation

After extraction, evaluate surrogate model similarity to the target:

```bash
python scripts/attacks/evaluate_surrogate_similarity.py \
  --surrogate_path /path/to/surrogate_model \
  --target_path /path/to/target_model \
  --test_data /path/to/test_data \
  --output_dir /path/to/output
```

Metrics include:
- Accuracy
- Agreement with target
- AUC-ROC
- Precision, Recall, F1-score
- Confusion matrices

## Documentation

- [docs/SETUP.md](docs/SETUP.md): Detailed setup and installation guide
- [docs/ORACLE_USAGE.md](docs/ORACLE_USAGE.md): Oracle query interface documentation
- [docs/BLACKBOX_COMPLIANCE.md](docs/BLACKBOX_COMPLIANCE.md): Black-box compliance and feature alignment
- [docs/reports/](docs/reports/): Generated evaluation reports and metrics

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This repository is part of the MIGATE framework. For issues, questions, or contributions, please refer to the main MIGATE repository or open an issue in this repository.
