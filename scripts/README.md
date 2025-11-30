# Prometheus Pipeline Scripts

This directory contains executable scripts for the Prometheus quantum phase discovery pipeline.

## Recreation Scripts

These scripts recreate all results from each version of the project:

### 2D Ising Model (Original Prometheus)
```bash
python scripts/recreate_2d_ising_results.py          # Full recreation
python scripts/recreate_2d_ising_results.py --quick  # Quick validation
```

### 3D Ising Model (Validation Extension)
```bash
python scripts/recreate_3d_ising_results.py          # Full recreation
python scripts/recreate_3d_ising_results.py --quick  # Quick validation
```

### Quantum Discovery Campaign (DTFIM)
```bash
python scripts/recreate_quantum_discovery_results.py          # Standard mode
python scripts/recreate_quantum_discovery_results.py --quick  # Quick validation
python scripts/recreate_quantum_discovery_results.py --full   # Publication quality
```

---

## Core Scripts

### Data Generation
- `generate_data.py` - Generate Ising model configurations
- `generate_3d_ising_dataset.py` - Generate 3D Ising datasets
- `generate_vae_training_data.py` - Prepare VAE training data
- `generate_potts_dataset.py` - Generate Potts model data
- `generate_xy_dataset.py` - Generate XY model data

### VAE Training
- `train_vae.py` - Train VAE model
- `train_3d_vae.py` - Train 3D VAE
- `train_enhanced_vae.py` - Train enhanced VAE with physics loss
- `train_physics_informed_2d_vae.py` - Physics-informed 2D VAE
- `train_physics_informed_3d_vae.py` - Physics-informed 3D VAE

### Analysis
- `analyze_latent_space.py` - Latent space analysis
- `extract_2d_critical_exponents.py` - 2D critical exponent extraction
- `extract_3d_critical_exponents.py` - 3D critical exponent extraction
- `extract_3d_latent_representations.py` - Extract latent representations
- `detect_3d_critical_temperature.py` - Critical temperature detection

### Pipeline & Validation
- `run_prometheus_pipeline.py` - Full pipeline orchestration
- `validate_pipeline.py` - Configuration validation
- `run_minimal_validation.py` - Quick validation
- `run_accuracy_validation_pipeline.py` - Accuracy validation
- `validate_multi_system.py` - Multi-system validation
- `validate_on_real_monte_carlo.py` - Monte Carlo validation
- `validate_prometheus_across_all_systems.py` - Cross-system validation

### Task Scripts (Quantum Discovery Campaign)
- `run_task10_dtfim_refinement.py` - DTFIM anomaly refinement
- `run_task11_dtfim_characterization.py` - DTFIM characterization
- `run_task12_secondary_refinement.py` - Secondary system refinement
- `run_month2_decision_point.py` - Month 2 decision analysis
- `run_task14_finite_size_scaling.py` - Finite-size scaling
- `run_task15_critical_exponents.py` - Critical exponent extraction
- `run_task16_entanglement_analysis.py` - Entanglement analysis
- `run_task17_cross_validation.py` - Cross-validation
- `run_task18_month3_decision.py` - Month 3 decision point
- `run_task19_physical_explanation.py` - Physical explanation
- `run_task20_experimental_relevance.py` - Experimental relevance
- `run_task21_nature_figures.py` - Publication figures

### Publication
- `create_data_repository.py` - Create data repository for publication
- `generate_publication_materials.py` - Generate publication materials
- `generate_main_results_figure.py` - Main results figure
- `generate_baseline_comparison_figure.py` - Baseline comparison
- `generate_campaign_overview_figures.py` - Campaign overview
- `generate_final_results_summary.py` - Final results summary
- `generate_graphical_abstract.py` - Graphical abstract
- `final_publication_validation.py` - Final validation for publication

### Utilities
- `inspect_h5.py` - Inspect HDF5 files
- `generate_visualizations.py` - Generate visualizations

## Usage

### Run Full Pipeline
```bash
python scripts/run_prometheus_pipeline.py --config config/default_config.yaml
```

### Run Specific Task
```bash
python scripts/run_task21_nature_figures.py
```

### Generate Publication Materials
```bash
python scripts/create_data_repository.py
python scripts/generate_publication_materials.py
```

## Output Structure

```
data/           # Generated datasets
models/         # Trained models
results/        # Analysis results
publication/    # Publication materials
```
