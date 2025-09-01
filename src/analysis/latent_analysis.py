"""
Latent Space Analysis Framework

This module provides utilities for encoding datasets through trained VAE models,
extracting latent coordinates, and visualizing the learned latent space with
temperature-based coloring to reveal phase structure.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import h5py
from tqdm import tqdm

from ..models.vae import ConvolutionalVAE
from ..data.preprocessing import IsingDataset
from ..utils.logging_utils import get_logger, LoggingContext


@dataclass
class LatentRepresentation:
    """
    Container for latent space representations and associated data.
    
    Attributes:
        z1: First latent dimension coordinates
        z2: Second latent dimension coordinates  
        temperatures: Temperature values for each configuration
        magnetizations: Magnetization values for each configuration
        energies: Energy values for each configuration
        reconstruction_errors: Per-sample reconstruction errors
        sample_indices: Original dataset indices for each sample
    """
    z1: np.ndarray
    z2: np.ndarray
    temperatures: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray
    reconstruction_errors: np.ndarray
    sample_indices: np.ndarray
    
    @property
    def latent_coords(self) -> np.ndarray:
        """Get latent coordinates as (N, 2) array."""
        return np.column_stack([self.z1, self.z2])
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the representation."""
        return len(self.z1)
    
    def get_temperature_mask(self, temp_range: Tuple[float, float]) -> np.ndarray:
        """
        Get boolean mask for samples within temperature range.
        
        Args:
            temp_range: (min_temp, max_temp) range
            
        Returns:
            Boolean mask array
        """
        return (self.temperatures >= temp_range[0]) & (self.temperatures <= temp_range[1])
    
    def filter_by_temperature(self, temp_range: Tuple[float, float]) -> 'LatentRepresentation':
        """
        Create new LatentRepresentation filtered by temperature range.
        
        Args:
            temp_range: (min_temp, max_temp) range
            
        Returns:
            Filtered LatentRepresentation
        """
        mask = self.get_temperature_mask(temp_range)
        
        return LatentRepresentation(
            z1=self.z1[mask],
            z2=self.z2[mask],
            temperatures=self.temperatures[mask],
            magnetizations=self.magnetizations[mask],
            energies=self.energies[mask],
            reconstruction_errors=self.reconstruction_errors[mask],
            sample_indices=self.sample_indices[mask]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of the latent representation."""
        return {
            'n_samples': self.n_samples,
            'temperature_range': (float(np.min(self.temperatures)), float(np.max(self.temperatures))),
            'latent_z1': {
                'mean': float(np.mean(self.z1)),
                'std': float(np.std(self.z1)),
                'min': float(np.min(self.z1)),
                'max': float(np.max(self.z1))
            },
            'latent_z2': {
                'mean': float(np.mean(self.z2)),
                'std': float(np.std(self.z2)),
                'min': float(np.min(self.z2)),
                'max': float(np.max(self.z2))
            },
            'reconstruction_error': {
                'mean': float(np.mean(self.reconstruction_errors)),
                'std': float(np.std(self.reconstruction_errors)),
                'min': float(np.min(self.reconstruction_errors)),
                'max': float(np.max(self.reconstruction_errors))
            }
        }


class LatentAnalyzer:
    """
    Main class for latent space analysis and visualization.
    
    Provides methods for encoding datasets through trained VAE models,
    extracting latent coordinates with proper memory management for large
    datasets, and creating publication-ready visualizations.
    """
    
    def __init__(self, 
                 trained_vae: ConvolutionalVAE,
                 device: Optional[torch.device] = None):
        """
        Initialize LatentAnalyzer with trained VAE model.
        
        Args:
            trained_vae: Trained ConvolutionalVAE model
            device: Device for computation (auto-detected if None)
        """
        self.vae = trained_vae
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger(__name__)
        
        # Move model to device and set to evaluation mode
        self.vae.to(self.device)
        self.vae.eval()
        
        self.logger.info(f"LatentAnalyzer initialized with device: {self.device}")
    
    def encode_dataset(self, 
                      dataloader: DataLoader,
                      max_batches: Optional[int] = None,
                      return_physics: bool = True) -> LatentRepresentation:
        """
        Encode complete dataset through VAE with batch processing and memory management.
        
        Args:
            dataloader: DataLoader for the dataset to encode
            max_batches: Maximum number of batches to process (None for all)
            return_physics: Whether to extract physical quantities
            
        Returns:
            LatentRepresentation containing encoded data
        """
        self.logger.info(f"Encoding dataset with {len(dataloader)} batches")
        
        # Initialize storage lists
        z1_list = []
        z2_list = []
        temperatures_list = []
        magnetizations_list = []
        energies_list = []
        recon_errors_list = []
        indices_list = []
        
        # Process batches
        with torch.no_grad():
            with LoggingContext("Dataset Encoding"):
                batch_count = 0
                
                for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Encoding batches")):
                    if max_batches is not None and batch_count >= max_batches:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch_data, (tuple, list)):
                        # Dataset returns (config, temperature, magnetization, energy)
                        configs = batch_data[0].to(self.device)
                        if len(batch_data) >= 4:
                            temps = batch_data[1].cpu().numpy()
                            mags = batch_data[2].cpu().numpy()
                            energies = batch_data[3].cpu().numpy()
                        else:
                            temps = np.zeros(configs.shape[0])
                            mags = np.zeros(configs.shape[0])
                            energies = np.zeros(configs.shape[0])
                    else:
                        # Dataset returns only configurations
                        configs = batch_data.to(self.device)
                        temps = np.zeros(configs.shape[0])
                        mags = np.zeros(configs.shape[0])
                        energies = np.zeros(configs.shape[0])
                    
                    # Encode through VAE
                    mu, logvar = self.vae.encode(configs)
                    
                    # Use mean of latent distribution for deterministic encoding
                    z = mu  # Shape: (batch_size, latent_dim)
                    
                    # Calculate reconstruction error
                    reconstruction = self.vae.decode(z)
                    recon_error = torch.mean((configs - reconstruction) ** 2, dim=[1, 2, 3])
                    
                    # Store results
                    z_np = z.cpu().numpy()
                    z1_list.append(z_np[:, 0])
                    z2_list.append(z_np[:, 1])
                    temperatures_list.append(temps)
                    magnetizations_list.append(mags)
                    energies_list.append(energies)
                    recon_errors_list.append(recon_error.cpu().numpy())
                    
                    # Store batch indices for tracking
                    batch_indices = np.arange(batch_idx * dataloader.batch_size, 
                                            batch_idx * dataloader.batch_size + configs.shape[0])
                    indices_list.append(batch_indices)
                    
                    batch_count += 1
                    
                    # Memory management for large datasets
                    if batch_count % 100 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all results
        latent_repr = LatentRepresentation(
            z1=np.concatenate(z1_list),
            z2=np.concatenate(z2_list),
            temperatures=np.concatenate(temperatures_list),
            magnetizations=np.concatenate(magnetizations_list),
            energies=np.concatenate(energies_list),
            reconstruction_errors=np.concatenate(recon_errors_list),
            sample_indices=np.concatenate(indices_list)
        )
        
        self.logger.info(f"Encoded {latent_repr.n_samples} samples")
        self.logger.info(f"Latent space statistics: {latent_repr.get_statistics()}")
        
        return latent_repr
    
    def encode_from_hdf5(self,
                        hdf5_path: str,
                        split: str = 'train',
                        batch_size: int = 128,
                        max_samples: Optional[int] = None) -> LatentRepresentation:
        """
        Encode dataset directly from HDF5 file with memory-efficient processing.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            split: Dataset split to encode ('train', 'val', 'test')
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process
            
        Returns:
            LatentRepresentation for the specified split
        """
        self.logger.info(f"Encoding {split} split from {hdf5_path}")
        
        # Create dataset and dataloader
        dataset = IsingDataset(hdf5_path, split=split, load_physics=True)
        
        # Limit dataset size if requested
        if max_samples is not None and max_samples < len(dataset):
            # Create subset indices
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset.indices = dataset.indices[indices]
            dataset.n_samples = len(indices)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return self.encode_dataset(dataloader, return_physics=True)
    
    def visualize_latent_space(self,
                              latent_repr: LatentRepresentation,
                              color_by: str = 'temperature',
                              figsize: Tuple[int, int] = (10, 8),
                              alpha: float = 0.6,
                              s: float = 20,
                              cmap: str = 'viridis') -> Figure:
        """
        Create scatter plot visualization of latent space with temperature coloring.
        
        Args:
            latent_repr: LatentRepresentation to visualize
            color_by: Variable to color points by ('temperature', 'magnetization', 'energy', 'reconstruction_error')
            figsize: Figure size (width, height)
            alpha: Point transparency
            s: Point size
            cmap: Colormap name
            
        Returns:
            Matplotlib Figure object
        """
        self.logger.info(f"Creating latent space visualization colored by {color_by}")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select coloring variable
        if color_by == 'temperature':
            color_values = latent_repr.temperatures
            color_label = 'Temperature'
        elif color_by == 'magnetization':
            color_values = np.abs(latent_repr.magnetizations)  # Use absolute magnetization
            color_label = '|Magnetization|'
        elif color_by == 'energy':
            color_values = latent_repr.energies
            color_label = 'Energy'
        elif color_by == 'reconstruction_error':
            color_values = latent_repr.reconstruction_errors
            color_label = 'Reconstruction Error'
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        # Create scatter plot
        scatter = ax.scatter(
            latent_repr.z1,
            latent_repr.z2,
            c=color_values,
            cmap=cmap,
            alpha=alpha,
            s=s,
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, fontsize=12)
        
        # Set labels and title
        ax.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax.set_title(f'Latent Space Representation\nColored by {color_label}', fontsize=14)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio to equal for proper visualization
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_phase_separation(self,
                                  latent_repr: LatentRepresentation,
                                  critical_temp: float = 2.269,
                                  temp_tolerance: float = 0.1,
                                  figsize: Tuple[int, int] = (12, 5)) -> Figure:
        """
        Create visualization showing phase separation in latent space.
        
        Args:
            latent_repr: LatentRepresentation to visualize
            critical_temp: Critical temperature for phase separation
            temp_tolerance: Temperature tolerance around critical point
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with phase separation plots
        """
        self.logger.info("Creating phase separation visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Define temperature regions
        low_temp_mask = latent_repr.temperatures < (critical_temp - temp_tolerance)
        high_temp_mask = latent_repr.temperatures > (critical_temp + temp_tolerance)
        critical_mask = ~(low_temp_mask | high_temp_mask)
        
        # Plot 1: Phase regions
        ax1.scatter(latent_repr.z1[low_temp_mask], latent_repr.z2[low_temp_mask],
                   c='blue', alpha=0.6, s=20, label=f'T < {critical_temp - temp_tolerance:.2f} (Ordered)')
        ax1.scatter(latent_repr.z1[high_temp_mask], latent_repr.z2[high_temp_mask],
                   c='red', alpha=0.6, s=20, label=f'T > {critical_temp + temp_tolerance:.2f} (Disordered)')
        ax1.scatter(latent_repr.z1[critical_mask], latent_repr.z2[critical_mask],
                   c='orange', alpha=0.8, s=25, label=f'Critical Region')
        
        ax1.set_xlabel('Latent Dimension 1 (z₁)')
        ax1.set_ylabel('Latent Dimension 2 (z₂)')
        ax1.set_title('Phase Separation in Latent Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot 2: Temperature gradient
        scatter = ax2.scatter(latent_repr.z1, latent_repr.z2,
                            c=latent_repr.temperatures, cmap='coolwarm',
                            alpha=0.6, s=20)
        
        # Add critical temperature contour if possible
        try:
            # Create temperature interpolation for contour
            from scipy.interpolate import griddata
            
            # Create grid for interpolation
            z1_grid = np.linspace(latent_repr.z1.min(), latent_repr.z1.max(), 50)
            z2_grid = np.linspace(latent_repr.z2.min(), latent_repr.z2.max(), 50)
            Z1_grid, Z2_grid = np.meshgrid(z1_grid, z2_grid)
            
            # Interpolate temperature
            temp_grid = griddata(
                (latent_repr.z1, latent_repr.z2),
                latent_repr.temperatures,
                (Z1_grid, Z2_grid),
                method='linear'
            )
            
            # Add critical temperature contour
            contour = ax2.contour(Z1_grid, Z2_grid, temp_grid, 
                                levels=[critical_temp], colors='black', linewidths=2)
            ax2.clabel(contour, inline=True, fontsize=10, fmt=f'T_c = {critical_temp:.3f}')
            
        except ImportError:
            self.logger.warning("scipy not available for contour plotting")
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Temperature')
        
        ax2.set_xlabel('Latent Dimension 1 (z₁)')
        ax2.set_ylabel('Latent Dimension 2 (z₂)')
        ax2.set_title('Temperature Distribution in Latent Space')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        return fig
    
    def save_latent_representation(self,
                                  latent_repr: LatentRepresentation,
                                  output_path: str) -> None:
        """
        Save LatentRepresentation to HDF5 file for later analysis.
        
        Args:
            latent_repr: LatentRepresentation to save
            output_path: Output file path
        """
        self.logger.info(f"Saving latent representation to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save latent coordinates
            f.create_dataset('z1', data=latent_repr.z1, compression='gzip')
            f.create_dataset('z2', data=latent_repr.z2, compression='gzip')
            
            # Save physical quantities
            f.create_dataset('temperatures', data=latent_repr.temperatures, compression='gzip')
            f.create_dataset('magnetizations', data=latent_repr.magnetizations, compression='gzip')
            f.create_dataset('energies', data=latent_repr.energies, compression='gzip')
            f.create_dataset('reconstruction_errors', data=latent_repr.reconstruction_errors, compression='gzip')
            f.create_dataset('sample_indices', data=latent_repr.sample_indices, compression='gzip')
            
            # Save metadata
            stats = latent_repr.get_statistics()
            metadata_group = f.create_group('metadata')
            for key, value in stats.items():
                if isinstance(value, dict):
                    subgroup = metadata_group.create_group(key)
                    for subkey, subvalue in value.items():
                        subgroup.attrs[subkey] = subvalue
                else:
                    metadata_group.attrs[key] = value
        
        self.logger.info(f"Latent representation saved: {output_path}")
    
    @staticmethod
    def load_latent_representation(file_path: str) -> LatentRepresentation:
        """
        Load LatentRepresentation from HDF5 file.
        
        Args:
            file_path: Path to saved latent representation file
            
        Returns:
            Loaded LatentRepresentation
        """
        with h5py.File(file_path, 'r') as f:
            return LatentRepresentation(
                z1=f['z1'][:],
                z2=f['z2'][:],
                temperatures=f['temperatures'][:],
                magnetizations=f['magnetizations'][:],
                energies=f['energies'][:],
                reconstruction_errors=f['reconstruction_errors'][:],
                sample_indices=f['sample_indices'][:]
            )
    
    def analyze_latent_dimensions(self, latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Analyze properties of individual latent dimensions.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            
        Returns:
            Dictionary with analysis results for each dimension
        """
        self.logger.info("Analyzing latent dimension properties")
        
        analysis = {}
        
        for dim_idx, (dim_name, dim_values) in enumerate([('z1', latent_repr.z1), ('z2', latent_repr.z2)]):
            dim_analysis = {}
            
            # Basic statistics
            dim_analysis['statistics'] = {
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'min': float(np.min(dim_values)),
                'max': float(np.max(dim_values)),
                'range': float(np.max(dim_values) - np.min(dim_values))
            }
            
            # Correlation with physical quantities
            temp_corr = np.corrcoef(dim_values, latent_repr.temperatures)[0, 1]
            mag_corr = np.corrcoef(dim_values, np.abs(latent_repr.magnetizations))[0, 1]
            energy_corr = np.corrcoef(dim_values, latent_repr.energies)[0, 1]
            
            dim_analysis['correlations'] = {
                'temperature': float(temp_corr),
                'abs_magnetization': float(mag_corr),
                'energy': float(energy_corr)
            }
            
            # Temperature dependence analysis
            unique_temps = np.unique(latent_repr.temperatures)
            temp_means = []
            temp_stds = []
            
            for temp in unique_temps:
                temp_mask = latent_repr.temperatures == temp
                temp_values = dim_values[temp_mask]
                temp_means.append(np.mean(temp_values))
                temp_stds.append(np.std(temp_values))
            
            dim_analysis['temperature_dependence'] = {
                'temperatures': unique_temps.tolist(),
                'means': temp_means,
                'stds': temp_stds,
                'mean_variation': float(np.std(temp_means)),
                'std_variation': float(np.std(temp_stds))
            }
            
            analysis[dim_name] = dim_analysis
        
        return analysis