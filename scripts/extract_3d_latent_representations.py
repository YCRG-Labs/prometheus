#!/usr/bin/env python3
"""
3D Latent Representation Extraction Script for Prometheus Project

This script implements task 5.2: Generate latent representations and compute order parameter correlations
- Extract latent representations for all 3D configurations
- Compute correlation between latent dimensions and magnetization
- Identify optimal latent dimension for order parameter representation
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vae_3d import ConvolutionalVAE3D
from src.data.data_loader_utils import AdaptiveDataLoader, DatasetFactory
from src.utils.config import PrometheusConfig


class LatentAnalyzer3D:
    """Analyzer for 3D VAE latent representations and order parameter correlations."""
    
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the latent analyzer.
        
        Args:
            model_path: Path to trained 3D VAE model
            device: Device to run analysis on
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> ConvolutionalVAE3D:
        """Load trained 3D VAE model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        
        # Create model with saved configuration
        model = ConvolutionalVAE3D(
            input_shape=model_config['input_shape'],
            latent_dim=model_config['latent_dim'],
            beta=model_config['beta']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def extract_latent_representations(self, dataloader: DataLoader) -> tuple:
        """
        Extract latent representations for all configurations in dataloader.
        
        Args:
            dataloader: DataLoader containing 3D configurations
            
        Returns:
            Tuple of (latent_representations, magnetizations, temperatures, energies)
        """
        latent_reps = []
        magnetizations = []
        temperatures = []
        energies = []
        
        print("Extracting latent representations...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    # If batch contains multiple elements (config, temp, mag, energy)
                    configs = batch[0].to(self.device)
                    if len(batch) > 1:
                        temps = batch[1].cpu().numpy()
                        temperatures.extend(temps)
                    if len(batch) > 2:
                        mags = batch[2].cpu().numpy()
                        magnetizations.extend(mags)
                    if len(batch) > 3:
                        engs = batch[3].cpu().numpy()
                        energies.extend(engs)
                else:
                    # Only configurations
                    configs = batch.to(self.device)
                
                # Extract latent representations (use mean of distribution)
                mu, logvar = self.model.encode(configs)
                latent_reps.append(mu.cpu().numpy())
                
                # If magnetizations not provided, compute from configurations
                if len(magnetizations) == 0:
                    # Convert from [-1, 1] spin values to magnetization
                    batch_mags = torch.mean(configs, dim=(1, 2, 3, 4)).cpu().numpy()
                    magnetizations.extend(batch_mags)
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
        
        # Concatenate all results
        latent_representations = np.concatenate(latent_reps, axis=0)
        magnetizations = np.array(magnetizations)
        
        if temperatures:
            temperatures = np.array(temperatures)
        else:
            temperatures = None
            
        if energies:
            energies = np.array(energies)
        else:
            energies = None
        
        print(f"Extracted {len(latent_representations)} latent representations")
        print(f"Latent space shape: {latent_representations.shape}")
        
        return latent_representations, magnetizations, temperatures, energies
    
    def compute_order_parameter_correlations(self, latent_reps: np.ndarray, 
                                           magnetizations: np.ndarray) -> dict:
        """
        Compute correlations between latent dimensions and magnetization.
        
        Args:
            latent_reps: Latent representations array (N, latent_dim)
            magnetizations: Magnetization values (N,)
            
        Returns:
            Dictionary with correlation results
        """
        print("\nComputing order parameter correlations...")
        
        correlations = {}
        latent_dim = latent_reps.shape[1]
        
        # Compute correlations for each latent dimension
        for dim in range(latent_dim):
            latent_values = latent_reps[:, dim]
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(latent_values, np.abs(magnetizations))
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = spearmanr(latent_values, np.abs(magnetizations))
            
            correlations[f'dim_{dim}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'abs_pearson_r': abs(pearson_r),
                'abs_spearman_r': abs(spearman_r)
            }
            
            print(f"  Dimension {dim}:")
            print(f"    Pearson r = {pearson_r:.4f} (p = {pearson_p:.4e})")
            print(f"    Spearman r = {spearman_r:.4f} (p = {spearman_p:.4e})")
        
        # Identify optimal dimension (highest absolute correlation)
        best_dim_pearson = max(range(latent_dim), 
                              key=lambda d: correlations[f'dim_{d}']['abs_pearson_r'])
        best_dim_spearman = max(range(latent_dim), 
                               key=lambda d: correlations[f'dim_{d}']['abs_spearman_r'])
        
        correlations['optimal_dimension'] = {
            'pearson_best': best_dim_pearson,
            'spearman_best': best_dim_spearman,
            'pearson_best_r': correlations[f'dim_{best_dim_pearson}']['abs_pearson_r'],
            'spearman_best_r': correlations[f'dim_{best_dim_spearman}']['abs_spearman_r']
        }
        
        print(f"\nOptimal dimensions:")
        print(f"  Pearson: Dimension {best_dim_pearson} (|r| = {correlations['optimal_dimension']['pearson_best_r']:.4f})")
        print(f"  Spearman: Dimension {best_dim_spearman} (|r| = {correlations['optimal_dimension']['spearman_best_r']:.4f})")
        
        return correlations
    
    def analyze_latent_space_structure(self, latent_reps: np.ndarray, 
                                     magnetizations: np.ndarray,
                                     temperatures: np.ndarray = None) -> dict:
        """
        Analyze the structure of the latent space.
        
        Args:
            latent_reps: Latent representations
            magnetizations: Magnetization values
            temperatures: Temperature values (optional)
            
        Returns:
            Dictionary with analysis results
        """
        print("\nAnalyzing latent space structure...")
        
        analysis = {}
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': np.mean(latent_reps, axis=0).tolist(),
            'std': np.std(latent_reps, axis=0).tolist(),
            'min': np.min(latent_reps, axis=0).tolist(),
            'max': np.max(latent_reps, axis=0).tolist()
        }
        
        # Correlation matrix between latent dimensions
        latent_corr = np.corrcoef(latent_reps.T)
        analysis['latent_correlations'] = latent_corr.tolist()
        
        # Variance explained by each dimension
        variances = np.var(latent_reps, axis=0)
        total_variance = np.sum(variances)
        variance_ratios = variances / total_variance
        analysis['variance_explained'] = {
            'variances': variances.tolist(),
            'variance_ratios': variance_ratios.tolist(),
            'cumulative_variance': np.cumsum(variance_ratios).tolist()
        }
        
        print(f"  Variance explained by each dimension: {variance_ratios}")
        print(f"  Cumulative variance: {np.cumsum(variance_ratios)}")
        
        # Temperature-dependent analysis if temperatures available
        if temperatures is not None:
            analysis['temperature_analysis'] = self._analyze_temperature_dependence(
                latent_reps, magnetizations, temperatures
            )
        
        return analysis
    
    def _analyze_temperature_dependence(self, latent_reps: np.ndarray,
                                      magnetizations: np.ndarray,
                                      temperatures: np.ndarray) -> dict:
        """Analyze temperature dependence of latent representations."""
        print("  Analyzing temperature dependence...")
        
        # Group by temperature
        unique_temps = np.unique(temperatures)
        temp_analysis = {}
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            temp_latent = latent_reps[temp_mask]
            temp_mag = magnetizations[temp_mask]
            
            temp_analysis[f'T_{temp:.3f}'] = {
                'mean_latent': np.mean(temp_latent, axis=0).tolist(),
                'std_latent': np.std(temp_latent, axis=0).tolist(),
                'mean_magnetization': float(np.mean(np.abs(temp_mag))),
                'std_magnetization': float(np.std(np.abs(temp_mag)))
            }
        
        return temp_analysis
    
    def create_visualization_plots(self, latent_reps: np.ndarray,
                                 magnetizations: np.ndarray,
                                 temperatures: np.ndarray = None,
                                 correlations: dict = None,
                                 output_dir: str = "results/3d_latent_analysis") -> dict:
        """
        Create visualization plots for latent space analysis.
        
        Args:
            latent_reps: Latent representations
            magnetizations: Magnetization values
            temperatures: Temperature values (optional)
            correlations: Correlation results
            output_dir: Output directory for plots
            
        Returns:
            Dictionary with plot file paths
        """
        print("\nCreating visualization plots...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Latent space scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by magnetization
        scatter = axes[0].scatter(latent_reps[:, 0], latent_reps[:, 1], 
                                c=np.abs(magnetizations), cmap='viridis', alpha=0.6)
        axes[0].set_xlabel('Latent Dimension 0')
        axes[0].set_ylabel('Latent Dimension 1')
        axes[0].set_title('Latent Space (colored by |Magnetization|)')
        plt.colorbar(scatter, ax=axes[0], label='|Magnetization|')
        
        # Color by temperature if available
        if temperatures is not None:
            scatter2 = axes[1].scatter(latent_reps[:, 0], latent_reps[:, 1], 
                                     c=temperatures, cmap='coolwarm', alpha=0.6)
            axes[1].set_xlabel('Latent Dimension 0')
            axes[1].set_ylabel('Latent Dimension 1')
            axes[1].set_title('Latent Space (colored by Temperature)')
            plt.colorbar(scatter2, ax=axes[1], label='Temperature')
        else:
            axes[1].hist2d(latent_reps[:, 0], latent_reps[:, 1], bins=50, cmap='Blues')
            axes[1].set_xlabel('Latent Dimension 0')
            axes[1].set_ylabel('Latent Dimension 1')
            axes[1].set_title('Latent Space Density')
        
        plt.tight_layout()
        latent_plot_path = Path(output_dir) / "latent_space_scatter.png"
        plt.savefig(latent_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['latent_scatter'] = str(latent_plot_path)
        
        # 2. Order parameter correlation plots
        if correlations is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for dim in range(latent_reps.shape[1]):
                axes[dim].scatter(latent_reps[:, dim], np.abs(magnetizations), alpha=0.6)
                axes[dim].set_xlabel(f'Latent Dimension {dim}')
                axes[dim].set_ylabel('|Magnetization|')
                
                # Add correlation info
                corr_info = correlations[f'dim_{dim}']
                axes[dim].set_title(f'Dim {dim} vs |Magnetization|\n'
                                  f'Pearson r = {corr_info["pearson_r"]:.4f}')
            
            plt.tight_layout()
            corr_plot_path = Path(output_dir) / "order_parameter_correlations.png"
            plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['correlations'] = str(corr_plot_path)
        
        # 3. Temperature evolution if available
        if temperatures is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            unique_temps = np.unique(temperatures)
            
            # Latent dimensions vs temperature
            temp_means = []
            temp_stds = []
            
            for temp in unique_temps:
                temp_mask = temperatures == temp
                temp_latent = latent_reps[temp_mask]
                temp_means.append(np.mean(temp_latent, axis=0))
                temp_stds.append(np.std(temp_latent, axis=0))
            
            temp_means = np.array(temp_means)
            temp_stds = np.array(temp_stds)
            
            for dim in range(latent_reps.shape[1]):
                axes[0, dim].errorbar(unique_temps, temp_means[:, dim], 
                                    yerr=temp_stds[:, dim], marker='o')
                axes[0, dim].set_xlabel('Temperature')
                axes[0, dim].set_ylabel(f'Latent Dimension {dim}')
                axes[0, dim].set_title(f'Latent Dim {dim} vs Temperature')
                axes[0, dim].grid(True, alpha=0.3)
            
            # Magnetization vs temperature
            mag_means = []
            mag_stds = []
            
            for temp in unique_temps:
                temp_mask = temperatures == temp
                temp_mag = np.abs(magnetizations[temp_mask])
                mag_means.append(np.mean(temp_mag))
                mag_stds.append(np.std(temp_mag))
            
            axes[1, 0].errorbar(unique_temps, mag_means, yerr=mag_stds, 
                              marker='o', color='red')
            axes[1, 0].set_xlabel('Temperature')
            axes[1, 0].set_ylabel('|Magnetization|')
            axes[1, 0].set_title('Magnetization vs Temperature')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add theoretical Tc line
            axes[1, 0].axvline(x=4.511, color='black', linestyle='--', 
                             label='Theoretical Tc = 4.511')
            axes[1, 0].legend()
            
            # Latent space evolution with temperature
            # Show how latent space changes with temperature
            temp_colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_temps)))
            
            for i, temp in enumerate(unique_temps[::2]):  # Show every other temperature
                temp_mask = temperatures == temp
                temp_latent = latent_reps[temp_mask]
                axes[1, 1].scatter(temp_latent[:, 0], temp_latent[:, 1], 
                                 c=[temp_colors[i*2]], alpha=0.6, 
                                 label=f'T = {temp:.2f}', s=20)
            
            axes[1, 1].set_xlabel('Latent Dimension 0')
            axes[1, 1].set_ylabel('Latent Dimension 1')
            axes[1, 1].set_title('Latent Space Evolution with Temperature')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            temp_plot_path = Path(output_dir) / "temperature_analysis.png"
            plt.savefig(temp_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['temperature'] = str(temp_plot_path)
        
        print(f"  Plots saved to: {output_dir}")
        return plot_paths


def load_3d_dataset_with_physics(hdf5_path: str) -> tuple:
    """Load 3D dataset with physical quantities."""
    
    class Physics3DDataset(torch.utils.data.Dataset):
        def __init__(self, hdf5_path, split='all'):
            self.hdf5_path = hdf5_path
            
            with h5py.File(hdf5_path, 'r') as f:
                if split == 'all':
                    # Load all data
                    self.configs = f['configurations'][:]
                    self.temperatures = f['temperatures'][:]
                    self.magnetizations = f['magnetizations'][:]
                    if 'energies' in f:
                        self.energies = f['energies'][:]
                    else:
                        self.energies = None
                else:
                    # Load specific split
                    indices = f[f'splits/{split}_indices'][:]
                    self.configs = f['configurations'][indices]
                    self.temperatures = f['temperatures'][indices]
                    self.magnetizations = f['magnetizations'][indices]
                    if 'energies' in f:
                        self.energies = f['energies'][indices]
                    else:
                        self.energies = None
                
        def __len__(self):
            return len(self.configs)
        
        def __getitem__(self, idx):
            config = torch.from_numpy(self.configs[idx]).float().unsqueeze(0)  # Add channel dim
            temp = self.temperatures[idx]
            mag = self.magnetizations[idx]
            
            if self.energies is not None:
                energy = self.energies[idx]
                return config, temp, mag, energy
            else:
                return config, temp, mag
    
    # Create dataset and dataloader
    dataset = Physics3DDataset(hdf5_path, 'all')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Extract 3D latent representations and analyze order parameters')
    parser.add_argument('--model', type=str, default='models/3d_vae/best_model.pth', 
                       help='Path to trained 3D VAE model')
    parser.add_argument('--data', type=str, default='data/ising_3d_small.h5',
                       help='Path to 3D HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='results/3d_latent_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("3D Latent Representation Analysis for Prometheus")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check if model exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Check if dataset exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    print("Loading trained 3D VAE model...")
    analyzer = LatentAnalyzer3D(args.model, device)
    
    # Load dataset with physics quantities
    print("Loading 3D dataset with physical quantities...")
    dataloader = load_3d_dataset_with_physics(args.data)
    
    # Extract latent representations
    start_time = time.time()
    latent_reps, magnetizations, temperatures, energies = analyzer.extract_latent_representations(dataloader)
    extraction_time = time.time() - start_time
    
    print(f"Extraction completed in {extraction_time:.2f} seconds")
    
    # Compute order parameter correlations
    correlations = analyzer.compute_order_parameter_correlations(latent_reps, magnetizations)
    
    # Analyze latent space structure
    analysis = analyzer.analyze_latent_space_structure(latent_reps, magnetizations, temperatures)
    
    # Create visualization plots
    plot_paths = analyzer.create_visualization_plots(
        latent_reps, magnetizations, temperatures, correlations, args.output_dir
    )
    
    # Save results
    results = {
        'extraction_info': {
            'model_path': args.model,
            'dataset_path': args.data,
            'extraction_time': extraction_time,
            'n_configurations': len(latent_reps),
            'latent_dimensions': latent_reps.shape[1]
        },
        'correlations': correlations,
        'latent_analysis': analysis,
        'plot_paths': plot_paths
    }
    
    # Save latent representations
    latent_data_path = Path(args.output_dir) / "latent_representations.npz"
    np.savez(
        latent_data_path,
        latent_representations=latent_reps,
        magnetizations=magnetizations,
        temperatures=temperatures if temperatures is not None else np.array([]),
        energies=energies if energies is not None else np.array([])
    )
    
    # Save analysis results
    import json
    results_path = Path(args.output_dir) / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("3D Latent Analysis Complete!")
    print("=" * 60)
    
    print(f"Results saved to: {args.output_dir}")
    print(f"Latent representations: {latent_data_path}")
    print(f"Analysis results: {results_path}")
    
    print(f"\nOrder Parameter Correlation Summary:")
    opt_dim = correlations['optimal_dimension']
    print(f"  Best dimension (Pearson): {opt_dim['pearson_best']} (|r| = {opt_dim['pearson_best_r']:.4f})")
    print(f"  Best dimension (Spearman): {opt_dim['spearman_best']} (|r| = {opt_dim['spearman_best_r']:.4f})")
    
    if temperatures is not None:
        print(f"\nTemperature range: {np.min(temperatures):.3f} - {np.max(temperatures):.3f}")
        print(f"Theoretical Tc: 4.511")
    
    print(f"\nVisualization plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")


if __name__ == "__main__":
    main()