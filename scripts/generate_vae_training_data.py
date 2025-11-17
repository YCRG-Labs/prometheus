#!/usr/bin/env python3
"""
Generate high-quality training data for VAE with strong magnetization signal.

The issue with current data is magnetization values are too small (~0.007).
We need data with temperatures well below Tc where magnetization is strong.
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def metropolis_3d_ising(L, T, n_steps, n_samples, sample_interval=100):
    """
    Simple but effective 3D Ising Monte Carlo simulation.
    
    Args:
        L: System size (L x L x L)
        T: Temperature
        n_steps: Equilibration steps
        n_samples: Number of configurations to sample
        sample_interval: Steps between samples
    """
    # Initialize with ordered state (all spins up) for T < Tc
    # This gives us a clear magnetization signal
    Tc = 4.511
    if T < Tc:
        spins = np.ones((L, L, L), dtype=np.int8)
    else:
        spins = 2 * np.random.randint(0, 2, size=(L, L, L)) - 1
    
    # Equilibration
    for _ in range(n_steps):
        # Random site
        i, j, k = np.random.randint(0, L, size=3)
        
        # Calculate energy change
        neighbors_sum = (
            spins[(i+1)%L, j, k] + spins[(i-1)%L, j, k] +
            spins[i, (j+1)%L, k] + spins[i, (j-1)%L, k] +
            spins[i, j, (k+1)%L] + spins[i, j, (k-1)%L]
        )
        dE = 2 * spins[i, j, k] * neighbors_sum
        
        # Metropolis acceptance
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j, k] *= -1
    
    # Sample configurations
    configurations = []
    magnetizations = []
    energies = []
    
    for _ in range(n_samples):
        # Monte Carlo steps between samples
        for _ in range(sample_interval):
            i, j, k = np.random.randint(0, L, size=3)
            neighbors_sum = (
                spins[(i+1)%L, j, k] + spins[(i-1)%L, j, k] +
                spins[i, (j+1)%L, k] + spins[i, (j-1)%L, k] +
                spins[i, j, (k+1)%L] + spins[i, j, (k-1)%L]
            )
            dE = 2 * spins[i, j, k] * neighbors_sum
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                spins[i, j, k] *= -1
        
        # Record configuration
        configurations.append(spins.copy())
        # Use absolute value of mean to get order parameter magnitude
        magnetizations.append(np.abs(np.mean(spins)))
        
        # Calculate energy
        energy = 0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    neighbors_sum = (
                        spins[(i+1)%L, j, k] + spins[i, (j+1)%L, k] + spins[i, j, (k+1)%L]
                    )
                    energy -= spins[i, j, k] * neighbors_sum
        energies.append(energy / (L**3))
    
    return np.array(configurations), np.array(magnetizations), np.array(energies)


def main():
    """Generate training data with strong magnetization signal."""
    
    print("="*60)
    print("Generating VAE Training Data")
    print("="*60)
    
    # Configuration
    L = 32  # System size
    Tc = 4.511  # Critical temperature for 3D Ising
    
    # Temperature schedule: focus on T < Tc where magnetization is strong
    # and around Tc for critical behavior
    temperatures = np.concatenate([
        np.linspace(3.0, 4.0, 15),  # Well below Tc: strong magnetization
        np.linspace(4.0, 4.8, 20),  # Around Tc: critical behavior
        np.linspace(4.8, 6.0, 15)   # Above Tc: disordered
    ])
    
    n_configs_per_temp = 30  # Configurations per temperature
    equilibration_steps = 20000  # Longer equilibration for better quality
    sample_interval = 200  # More steps between samples for independence
    
    print(f"System size: {L}x{L}x{L}")
    print(f"Temperatures: {len(temperatures)} from {temperatures[0]:.2f} to {temperatures[-1]:.2f}")
    print(f"Tc = {Tc:.3f}")
    print(f"Configurations per temperature: {n_configs_per_temp}")
    print(f"Total configurations: {len(temperatures) * n_configs_per_temp}")
    print()
    
    # Generate data
    all_configurations = []
    all_magnetizations = []
    all_energies = []
    all_temperatures = []
    
    for T in tqdm(temperatures, desc="Generating data"):
        configs, mags, energies = metropolis_3d_ising(
            L, T, equilibration_steps, n_configs_per_temp, sample_interval
        )
        
        all_configurations.extend(configs)
        all_magnetizations.extend(mags)
        all_energies.extend(energies)
        all_temperatures.extend([T] * n_configs_per_temp)
    
    # Convert to arrays
    all_configurations = np.array(all_configurations, dtype=np.int8)
    all_magnetizations = np.array(all_magnetizations, dtype=np.float32)
    all_energies = np.array(all_energies, dtype=np.float32)
    all_temperatures = np.array(all_temperatures, dtype=np.float32)
    
    print()
    print("Data Statistics:")
    print(f"  Configurations shape: {all_configurations.shape}")
    print(f"  Magnetization range: [{np.min(all_magnetizations):.4f}, {np.max(all_magnetizations):.4f}]")
    print(f"  Magnetization mean: {np.mean(all_magnetizations):.4f}")
    print(f"  Energy range: [{np.min(all_energies):.4f}, {np.max(all_energies):.4f}]")
    print(f"  Temperature range: [{np.min(all_temperatures):.2f}, {np.max(all_temperatures):.2f}]")
    
    # Save to HDF5
    output_path = Path("data/vae_training_3d_ising.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('configurations', data=all_configurations, compression='gzip')
        f.create_dataset('magnetizations', data=all_magnetizations)
        f.create_dataset('energies', data=all_energies)
        f.create_dataset('temperatures', data=all_temperatures)
        
        # Metadata
        f.attrs['system_size'] = L
        f.attrs['tc_theoretical'] = Tc
        f.attrs['n_temperatures'] = len(temperatures)
        f.attrs['n_configs_per_temp'] = n_configs_per_temp
        f.attrs['equilibration_steps'] = equilibration_steps
        f.attrs['sample_interval'] = sample_interval
    
    print()
    print(f"Data saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
