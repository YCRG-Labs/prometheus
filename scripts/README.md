# Prometheus Pipeline Scripts

This directory contains the executable scripts for running the Prometheus pipeline stages.

## Scripts Overview

### Individual Stage Scripts

1. **`generate_data.py`** - Data generation and preprocessing
   - Generates Ising model configurations across temperature ranges
   - Preprocesses data into HDF5 format for training
   - Supports parallel processing and validation

2. **`train_vae.py`** - VAE model training
   - Trains Convolutional VAE on Ising model data
   - Supports checkpointing, early stopping, and resumption
   - Configurable hyperparameters and training callbacks

3. **`analyze_latent_space.py`** - Post-training analysis
   - Encodes data into latent space using trained VAE
   - Discovers order parameters and detects phase transitions
   - Performs clustering and correlation analysis

4. **`generate_visualizations.py`** - Visualization generation
   - Creates publication-ready figures and plots
   - Generates analysis reports and figure indices
   - Supports multiple output formats (PNG, PDF, SVG)

### Pipeline Orchestration

5. **`run_prometheus_pipeline.py`** - Complete pipeline orchestrator
   - Runs the full end-to-end pipeline
   - Supports state management and resumption
   - Handles error recovery and dependency checking

6. **`validate_pipeline.py`** - Configuration validation utility
   - Validates pipeline configuration files
   - Checks system dependencies and requirements
   - Provides detailed error reporting

## Usage Examples

### Run Individual Stages

```bash
# Generate data
python scripts/generate_data.py --config config/default_config.yaml --output-dir data --parallel

# Train VAE
python scripts/train_vae.py --data data/processed_dataset.h5 --config config/default_config.yaml

# Analyze latent space
python scripts/analyze_latent_space.py --model models/final_model.pth --data data/processed_dataset.h5

# Generate visualizations
python scripts/generate_visualizations.py --analysis-results results/analysis_results.json --detailed-data results/detailed_analysis.npz
```

### Run Complete Pipeline

```bash
# Run full pipeline
python scripts/run_prometheus_pipeline.py --config config/default_config.yaml

# Resume interrupted pipeline
python scripts/run_prometheus_pipeline.py --resume

# Run specific stages
python scripts/run_prometheus_pipeline.py --stages data_generation vae_training

# Force re-run stages
python scripts/run_prometheus_pipeline.py --force data_generation --stages data_generation vae_training

# Check pipeline status
python scripts/run_prometheus_pipeline.py --status
```

### Validate Configuration

```bash
# Validate configuration file
python scripts/validate_pipeline.py --config config/my_config.yaml --check-deps --verbose

# Validate default configuration
python scripts/validate_pipeline.py --check-deps
```

## Configuration

All scripts use the same configuration system based on YAML files. See `config/default_config.yaml` for the complete configuration structure.

Key configuration sections:
- `ising`: Ising model simulation parameters
- `vae`: VAE architecture parameters  
- `training`: Training pipeline parameters
- `logging`: Logging configuration

## Pipeline State Management

The pipeline orchestrator maintains state in `.prometheus_pipeline_state.json` to support:
- Resumption after interruption
- Skipping completed stages
- Error recovery and retry
- Progress tracking

## Error Handling

All scripts include comprehensive error handling:
- Configuration validation before execution
- Dependency checking and verification
- Graceful failure with informative error messages
- State preservation for recovery

## Output Structure

The pipeline creates the following output structure:

```
data/                    # Generated and preprocessed data
├── raw_configurations/  # Raw simulation data
└── processed_dataset.h5 # Preprocessed training data

models/                  # Trained models and checkpoints
├── final_model.pth     # Final trained VAE model
├── checkpoints/        # Training checkpoints
└── training_history.json

results/                 # Analysis results and visualizations
├── analysis_results.json    # Quantitative analysis results
├── detailed_analysis.npz    # Detailed data arrays
├── latent_space.png        # Latent space visualization
├── order_parameter.png     # Order parameter plot
├── phase_diagram.png       # Phase diagram
├── analysis_report.md      # Comprehensive report
└── figure_index.md         # Figure index
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, scikit-learn
- Matplotlib
- h5py
- PyYAML
- tqdm

Run `python scripts/validate_pipeline.py --check-deps` to verify all dependencies are installed.