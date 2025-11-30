#!/usr/bin/env python3
"""
Recreate 3D Ising Model Results (Validation Extension)

This script recreates all results from the 3D Ising model validation,
demonstrating the framework's ability to detect phase transitions in
higher-dimensional systems.

Steps:
1. Generate 3D Ising configurations across temperature range
2. Train 3D VAE on configurations
3. Analyze latent space for phase transition detection
4. Extract critical temperature and validate against Tc ≈ 4.5115

Usage:
    python scripts/recreate_3d_ising_results.py
    python scripts/recreate_3d_ising_results.py --quick  # Fast validation
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_3d_ising_pipeline(quick_mode: bool = False, output_dir: str = None):
    """Run the complete 3D Ising model pipeline."""
    
    print("=" * 70)
    print("RECREATING 3D ISING MODEL RESULTS")
    print("Prometheus VAE Phase Detection - 3D Extension")
    print("=" * 70)
    print()
    
    # Set parameters based on mode
    if quick_mode:
        lattice_size = 8
        n_temps = 15
        n_configs = 50
        n_epochs = 15
        print("Running in QUICK mode (reduced parameters for validation)")
    else:
        lattice_size = 16
        n_temps = 40
        n_configs = 200
        n_epochs = 50
        print("Running in FULL mode (publication-quality results)")
    
    if output_dir is None:
        output_dir = f"results/3d_ising_recreation_{datetime.now():%Y%m%d_%H%M%S}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path}")
    print(f"Lattice size: {lattice_size}x{lattice_size}x{lattice_size}")
    print(f"Temperature points: {n_temps}")
    print(f"Configurations per temperature: {n_configs}")
    print()
    
    # Import modules
    try:
        import torch
        import numpy as np
        from src.data.ising_simulator import IsingSimulator3D
        from src.models.vae import ConvVAE3D
        from src.training.trainer import VAETrainer
        from src.analysis.phase_detector import PhaseDetector
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)
    
    # Theoretical critical temperature for 3D Ising
    Tc_exact = 4.5115  # High-precision Monte Carlo result
    print(f"Theoretical Tc = {Tc_exact:.4f}")
    print()
    
    # Step 1: Generate Data
    print("-" * 50)
    print("Step 1: Generating 3D Ising configurations")
    print("-" * 50)
    
    try:
        simulator = IsingSimulator3D(
            lattice_size=(lattice_size, lattice_size, lattice_size),
            boundary_conditions='periodic'
        )
    except:
        # Fallback to generic simulator
        from src.data.ising_simulator import IsingSimulator
        simulator = IsingSimulator(
            lattice_size=(lattice_size, lattice_size, lattice_size),
            boundary_conditions='periodic'
        )
    
    T_min, T_max = 3.5, 5.5
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    all_configs = []
    all_temps = []
    all_mags = []
    
    for i, T in enumerate(temperatures):
        print(f"  T = {T:.3f} ({i+1}/{n_temps})", end='\r')
        configs, mags = simulator.generate_configurations(
            temperature=T,
            n_configs=n_configs,
            equilibration_steps=2000,
            sampling_interval=20
        )
        all_configs.append(configs)
        all_temps.extend([T] * n_configs)
        all_mags.extend(mags)
    
    print()
    
    configs_array = np.concatenate(all_configs, axis=0)
    temps_array = np.array(all_temps)
    mags_array = np.array(all_mags)
    
    print(f"  Generated {len(configs_array)} configurations")
    
    # Save raw data
    data_file = output_path / "ising_3d_data.npz"
    np.savez(data_file, 
             configurations=configs_array,
             temperatures=temps_array,
             magnetizations=mags_array)
    print(f"  Saved to: {data_file}")
    
    # Step 2: Train VAE
    print()
    print("-" * 50)
    print("Step 2: Training 3D VAE")
    print("-" * 50)
    
    # Prepare data
    X = configs_array.reshape(-1, 1, lattice_size, lattice_size, lattice_size).astype(np.float32)
    X = (X + 1) / 2  # Convert from {-1, 1} to {0, 1}
    
    # Create 3D VAE
    try:
        vae = ConvVAE3D(
            input_channels=1,
            input_size=lattice_size,
            latent_dim=2,
            hidden_dims=[16, 32, 64]
        )
    except:
        # Fallback to 2D VAE with flattened input
        print("  Note: Using adapted 2D VAE for 3D data")
        from src.models.vae import ConvVAE
        vae = ConvVAE(
            input_channels=lattice_size,  # Treat z-slices as channels
            input_size=lattice_size,
            latent_dim=2,
            hidden_dims=[32, 64, 128]
        )
        X = X.reshape(-1, lattice_size, lattice_size, lattice_size)
    
    # Train
    trainer = VAETrainer(
        model=vae,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"  Device: {trainer.device}")
    print(f"  Training for {n_epochs} epochs...")
    
    history = trainer.train(
        X, 
        epochs=n_epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=True
    )
    
    # Save model
    model_file = output_path / "vae_3d_ising.pth"
    torch.save(vae.state_dict(), model_file)
    print(f"  Model saved to: {model_file}")
    
    # Step 3: Analyze Latent Space
    print()
    print("-" * 50)
    print("Step 3: Analyzing latent space")
    print("-" * 50)
    
    # Encode all data
    vae.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(trainer.device)
        z_mean, z_logvar = vae.encode(X_tensor)
        latent_coords = z_mean.cpu().numpy()
    
    print(f"  Latent space shape: {latent_coords.shape}")
    
    # Detect phase transition
    detector = PhaseDetector()
    
    # Find critical temperature from latent space
    Tc_detected = detector.find_critical_temperature(
        latent_coords, 
        temps_array,
        method='susceptibility'
    )
    
    error = abs(Tc_detected - Tc_exact) / Tc_exact * 100
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Detected Tc:    {Tc_detected:.4f}")
    print(f"  Theoretical Tc: {Tc_exact:.4f}")
    print(f"  Relative Error: {error:.2f}%")
    print()
    
    if error < 5:
        print("  ✓ SUCCESS: Error < 5%")
    elif error < 10:
        print("  ⚠ ACCEPTABLE: Error < 10%")
    else:
        print("  ✗ WARNING: Error > 10%")
    
    # Save results
    results = {
        'Tc_detected': float(Tc_detected),
        'Tc_theoretical': float(Tc_exact),
        'relative_error_percent': float(error),
        'lattice_size': lattice_size,
        'n_temperatures': n_temps,
        'n_configs_per_temp': n_configs,
        'n_epochs': n_epochs,
        'quick_mode': quick_mode,
        'dimension': 3
    }
    
    import json
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    # Generate visualization
    print()
    print("-" * 50)
    print("Step 4: Generating visualizations")
    print("-" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Latent space colored by temperature
        scatter = axes[0].scatter(
            latent_coords[:, 0], latent_coords[:, 1],
            c=temps_array, cmap='coolwarm', alpha=0.5, s=1
        )
        axes[0].set_xlabel('Latent dim 1')
        axes[0].set_ylabel('Latent dim 2')
        axes[0].set_title('Latent Space (colored by T)')
        plt.colorbar(scatter, ax=axes[0], label='Temperature')
        
        # Magnetization vs temperature
        unique_temps = np.unique(temps_array)
        mean_mags = [np.mean(np.abs(mags_array[temps_array == T])) for T in unique_temps]
        axes[1].plot(unique_temps, mean_mags, 'o-')
        axes[1].axvline(Tc_exact, color='r', linestyle='--', label=f'Tc (exact) = {Tc_exact:.3f}')
        axes[1].axvline(Tc_detected, color='g', linestyle=':', label=f'Tc (detected) = {Tc_detected:.3f}')
        axes[1].set_xlabel('Temperature')
        axes[1].set_ylabel('|Magnetization|')
        axes[1].set_title('Order Parameter')
        axes[1].legend()
        
        # Training loss
        axes[2].plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            axes[2].plot(history['val_loss'], label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training History')
        axes[2].legend()
        
        plt.tight_layout()
        fig_file = output_path / "3d_ising_results.png"
        plt.savefig(fig_file, dpi=150)
        plt.close()
        print(f"  Figure saved to: {fig_file}")
        
    except Exception as e:
        print(f"  Warning: Could not generate visualization: {e}")
    
    print()
    print("=" * 70)
    print("3D ISING MODEL RECREATION COMPLETE")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recreate 3D Ising model results"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run in quick mode with reduced parameters'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    results = run_3d_ising_pipeline(
        quick_mode=args.quick,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if results['relative_error_percent'] < 15 else 1)


if __name__ == "__main__":
    main()
