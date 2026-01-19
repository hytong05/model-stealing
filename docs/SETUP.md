# Setup Guide

This guide provides detailed instructions for setting up the MIR (Model Inferred Replica) component of the MIGATE framework.

## System Requirements

- Python 3.8 or higher
- pip (usually included with Python)
- Git (for installing ember package)

## Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup_venv.sh
```

This script will:
1. Check Python version
2. Create a virtual environment in `venv/`
3. Install all dependencies from `requirements.txt`
4. Install the `ember` package from GitHub (not available on PyPI)

## Manual Setup

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install ember package from GitHub (not available on PyPI)
pip install git+https://github.com/endgameinc/ember.git
```

## Usage

After setup, each time you work with the project:

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run scripts:**
   ```bash
   python scripts/attacks/model_extraction.py --help
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

## Project Structure

```
model-stealing/
├── src/                    # Core source code
│   ├── models/            # Model definitions (DNN, SOREL networks)
│   ├── attackers/         # Surrogate model attackers
│   ├── targets/           # Target model wrappers
│   ├── datasets/          # Dataset loaders
│   ├── utils/             # Utility functions
│   └── sampling/          # Active learning sampling strategies
├── scripts/               # Executable scripts
│   ├── attacks/           # Model extraction pipelines
│   ├── oracle/            # Oracle query interfaces
│   ├── data/              # Data preprocessing
│   ├── inference/         # Model inference utilities
│   └── examples/          # Example usage scripts
├── data/                  # Data files (not tracked in git)
├── artifacts/             # Model artifacts (not tracked)
│   └── targets/           # Target model files
├── output/                # Experiment outputs (not tracked)
├── storage/               # Training checkpoints (not tracked)
├── logs/                  # Log files (not tracked)
├── config/                # Configuration files
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
├── examples/              # Example scripts
├── venv/                  # Virtual environment (auto-created)
├── requirements.txt       # Python dependencies
└── setup_venv.sh         # Setup script
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
- Virtual environment is activated
- You are in the project root directory
- All dependencies are installed

### TensorFlow Installation Issues

If you have problems with TensorFlow, try:
```bash
pip install tensorflow --upgrade
```

### PyTorch Installation Issues

PyTorch may require separate installation depending on your OS and GPU. See: https://pytorch.org/get-started/locally/

### Ember Package Installation Issues

The `ember` package is not available on PyPI and must be installed from GitHub. If you encounter errors, try:
```bash
pip install tqdm lief
pip install git+https://github.com/endgameinc/ember.git
```

### Virtual Environment Issues

If the virtual environment doesn't activate:
- Ensure Python 3.8+ is installed
- Check that `venv/bin/activate` exists (Linux/Mac) or `venv\Scripts\activate` exists (Windows)
- Try recreating the virtual environment

## Next Steps

After setup, you can:
1. Read the [README.md](../README.md) for an overview
2. Check [ORACLE_USAGE.md](ORACLE_USAGE.md) for oracle query examples
3. Review [BLACKBOX_COMPLIANCE.md](BLACKBOX_COMPLIANCE.md) for black-box attack guidelines
4. Run example extraction attacks using `scripts/attacks/model_extraction.py`
