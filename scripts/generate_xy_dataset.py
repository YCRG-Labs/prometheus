#!/usr/bin/env python3
"""
Generate 2D XY Model Dataset and Apply Prometheus Analysis.

This script generates a comprehensive dataset for the 2D XY model,
applies the Prometheus VAE to learn representations, and detects the
Kosterlitz-Thouless (KT) topological transition at Tc ≈ 0.893/J.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.unified_monte_carlo import create_xy_simulator
from models.physics_models import create_xy_2d_model
from models.adaptive_vae import AdaptiveVAE
from training.trainer import VAETrainer
from analysis.critical_exponent_analyzer import CriticalExponentAnalyzer
from utils.visualization import create_phase_plots, save_figure


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/xy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_xy_data(lattice_size: Tuple[int, int],
                    temperature_range: Tuple[float, float],
                    n_temperatures: int = 50,
                    n_configs_per_temp: int = 200,
                    coupling_strength: float = 1.0,
                    equilibration_steps: int = 50000,
                    sampling_interval: int = 100) -> Dict[str, Any]:
    """
    Generate 2D XY model dataset.
    
    Args:
        lattice_size: 2D lattice dimensions (height, width)
        temperature_range: (T_min, T_max) temperature range
        n_temperatures: Number of temperature points
        n_configs_per_temp: Configurations per temperature
        coupling_strength: Coupling constant J
        equilibration_steps: Equilibration steps per temperature
        sampling_interval: Steps between samples
        
    Returns:
        Dictionary with generated data and metadata
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating 2D XY data: lattice_size={lattice_size}, "
               f"T_range={temperature_range}, n_temps={n_temperatures}")
    
    # Create XY simulator
    simulator = create_xy_simulator(
        lattice_size=lattice_size,
        temperature=1.0,  # Will be changed during simulation
        coupling_strength=coupling_strength
    )
    
    # Generate temperature series data
    result = simulator.simulate_temperature_series(
        temperature_range=temperature_range,
        n_temperatures=n_temperatures,
        n_configs_per_temp=n_configs_per_temp,
        sampling_interval=sampling_interval,
        equilibration_steps=equilibration_steps
    )
    
    logger.info(f"Generated {len(result.configurations)} total configurations")
    
    # Calculate additional properties specific to XY model
    logger.info("Computing XY model specific properties...")
    
    # Group data by temperature for analysis
    temperatures = np.unique(result.temperatures)
    temp_data = {}
    
    xy_model = create_xy_2d_model(coupling_strength)
    
    for temp in temperatures:
        mask = result.temperatures == temp
        temp_configs = result.configurations[mask]
        temp_order_params = result.order_parameters[mask]
        temp_energies = result.energies[mask]
        
        # Calculate XY-specific properties
        temp_vorticities = []
        temp_helicity_moduli = []
        
        for config in temp_configs:
            # Calculate vorticity (topological charge)
            vorticity = xy_model.compute_vorticity(config)
            temp_vorticities.append(vorticity)
            
            # Calculate helicity modulus (superfluid density)
            helicity_modulus = calculate_helicity_modulus(config, xy_model)
            temp_helicity_moduli.append(helicity_modulus)
        
        # Calculate statistics
        mean_order_param = np.mean(temp_order_params)
        std_order_param = np.std(temp_order_params)
        mean_energy = np.mean(temp_energies)
        std_energy = np.std(temp_energies)
        mean_vorticity = np.mean(temp_vorticities)
        std_vorticity = np.std(temp_vorticities)
        mean_helicity = np.mean(temp_helicity_moduli)
        std_helicity = np.std(temp_helicity_moduli)
        
        # Calculate susceptibility
        susceptibility = len(temp_order_params) * (np.mean(temp_order_params**2) - mean_order_param**2) / temp
        
        # Calculate specific heat
        specific_heat = (np.mean(temp_energies**2) - mean_energy**2) / (temp**2)
        
        temp_data[temp] = {
            'configurations': temp_configs,
            'order_parameters': temp_order_params,
            'energies': temp_energies,
            'vorticities': np.array(temp_vorticities),
            'helicity_moduli': np.array(temp_helicity_moduli),
            'mean_order_param': mean_order_param,
            'std_order_param': std_order_param,
            'mean_energy': mean_energy,
            'std_energy': std_energy,
            'mean_vorticity': mean_vorticity,
            'std_vorticity': std_vorticity,
            'mean_helicity': mean_helicity,
            'std_helicity': std_helicity,
            'susceptibility': susceptibility,
            'specific_heat': specific_heat
        }
    
    return {
        'configurations': result.configurations,
        'temperatures': result.temperatures,
        'order_parameters': result.order_parameters,
        'energies': result.energies,
        'temperature_data': temp_data,
        'model_info': result.model_info,
        'simulation_metadata': result.simulation_metadata,
        'unique_temperatures': temperatures
    }


def calculate_helicity_modulus(configuration: np.ndarray, xy_model) -> float:
    """
    Calculate helicity modulus (superfluid density) for XY configuration.
    
    The helicity modulus is related to the response to twisted boundary conditions
    and serves as an order parameter for the KT transition.
    
    Args:
        configuration: 2D angle configuration
        xy_model: XY model instance
        
    Returns:
        Helicity modulus value
    """
    height, width = configuration.shape
    
    # Calculate current-current correlation
    # This is a simplified calculation - full helicity modulus requires more complex analysis
    
    # Calculate local currents (simplified version)
    j_x = np.zeros_like(configuration)
    j_y = np.zeros_like(configuration)
    
    for i in range(height):
        for j in range(width):
            # Current in x-direction
            theta_right = configuration[i, (j + 1) % width]
            theta_current = configuration[i, j]
            j_x[i, j] = np.sin(theta_right - theta_current)
            
            # Current in y-direction
            theta_down = configuration[(i + 1) % height, j]
            j_y[i, j] = np.sin(theta_down - theta_current)
    
    # Helicity modulus approximation
    helicity = np.mean(j_x**2) + np.mean(j_y**2)
    
    return helicity


def analyze_kt_transition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze Kosterlitz-Thouless transition properties of XY model data.
    
    Args:
        data: Generated XY model data
        
    Returns:
        Dictionary with KT transition analysis results
    """
    logger = logging.getLogger(__name__)
    
    temperatures = data['unique_temperatures']
    temp_data = data['temperature_data']
    
    # Extract thermodynamic quantities
    mean_order_params = [temp_data[T]['mean_order_param'] for T in temperatures]
    susceptibilities = [temp_data[T]['susceptibility'] for T in temperatures]
    specific_heats = [temp_data[T]['specific_heat'] for T in temperatures]
    mean_helicity_moduli = [temp_data[T]['mean_helicity'] for T in temperatures]
    mean_vorticities = [temp_data[T]['mean_vorticity'] for T in temperatures]
    
    # For KT transition, look for jump in helicity modulus
    # Find temperature where helicity modulus drops significantly
    helicity_gradient = np.gradient(mean_helicity_moduli, temperatures)
    kt_transition_idx = np.argmin(helicity_gradient)  # Steepest drop
    tc_helicity = temperatures[kt_transition_idx]
    
    # Alternative: look for vortex unbinding (increase in vorticity variance)
    vorticity_variances = [temp_data[T]['std_vorticity']**2 for T in temperatures]
    max_vorticity_var_idx = np.argmax(vorticity_variances)
    tc_vorticity = temperatures[max_vorticity_var_idx]
    
    # Theoretical KT transition temperature
    tc_theoretical = data['model_info']['theoretical_tc']
    
    logger.info(f"Kosterlitz-Thouless transition analysis:")
    logger.info(f"  From helicity modulus: Tc = {tc_helicity:.4f}")
    logger.info(f"  From vorticity analysis: Tc = {tc_vorticity:.4f}")
    logger.info(f"  Theoretical value: Tc = {tc_theoretical:.4f}")
    
    # Use helicity modulus as primary estimate
    tc_measured = tc_helicity
    tc_error = abs(tc_measured - tc_theoretical) / tc_theoretical * 100
    
    logger.info(f"  KT transition temperature accuracy: {100 - tc_error:.1f}% "
               f"(error: {tc_error:.1f}%)")
    
    return {
        'tc_helicity': tc_helicity,
        'tc_vorticity': tc_vorticity,
        'tc_theoretical': tc_theoretical,
        'tc_measured': tc_measured,
        'tc_error_percent': tc_error,
        'temperatures': temperatures,
        'mean_order_params': np.array(mean_order_params),
        'susceptibilities': np.array(susceptibilities),
        'specific_heats': np.array(specific_heats),
        'mean_helicity_moduli': np.array(mean_helicity_moduli),
        'mean_vorticities': np.array(mean_vorticities),
        'vorticity_variances': np.array(vorticity_variances)
    }


def train_xy_vae(data: Dict[str, Any], 
                latent_dim: int = 2,
                epochs: int = 100,
                batch_size: int = 64) -> Tuple[AdaptiveVAE, Dict[str, Any]]:
    """
    Train VAE on XY model configurations.
    
    Args:
        data: Generated XY model data
        latent_dim: Latent space dimensionality
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Tuple of (trained_vae, training_results)
    """
    logger = logging.getLogger(__name__)
    
    configurations = data['configurations']
    temperatures = data['temperatures']
    
    logger.info(f"Training VAE on {len(configurations)} XY configurations")
    logger.info(f"Configuration shape: {configurations[0].shape}")
    
    # Prepare data for VAE
    # Convert angles to (cos, sin) representation for better VAE training
    n_configs, height, width = configurations.shape
    
    # Create 2-channel input: (cos(θ), sin(θ))
    vae_input = np.zeros((n_configs, 2, height, width))
    vae_input[:, 0, :, :] = np.cos(configurations)  # cos channel
    vae_input[:, 1, :, :] = np.sin(configurations)  # sin channel
    
    logger.info(f"VAE input shape: {vae_input.shape}")
    
    # Create adaptive VAE
    vae = AdaptiveVAE(
        input_shape=(2, height, width),
        latent_dim=latent_dim,
        beta=1.0  # Standard β-VAE
    )
    
    # Create trainer
    trainer = VAETrainer(
        model=vae,
        learning_rate=1e-3,
        device='cpu'  # Use CPU for compatibility
    )
    
    # Train VAE
    training_results = trainer.train(
        train_data=vae_input,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    
    logger.info(f"VAE training completed. Final loss: {training_results['final_loss']:.4f}")
    
    return vae, training_results


def extract_xy_latent_representations(vae: AdaptiveVAE, 
                                    data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract latent representations from trained XY VAE.
    
    Args:
        vae: Trained VAE model
        data: XY model data
        
    Returns:
        Dictionary with latent representations and analysis
    """
    logger = logging.getLogger(__name__)
    
    configurations = data['configurations']
    temperatures = data['temperatures']
    order_parameters = data['order_parameters']
    
    # Prepare configurations (same as training)
    n_configs, height, width = configurations.shape
    
    vae_input = np.zeros((n_configs, 2, height, width))
    vae_input[:, 0, :, :] = np.cos(configurations)
    vae_input[:, 1, :, :] = np.sin(configurations)
    
    logger.info("Extracting latent representations...")
    
    # Extract latent representations
    latent_representations = vae.encode(vae_input)
    
    logger.info(f"Extracted latent representations: shape {latent_representations.shape}")
    
    # Analyze correlations with physical quantities
    correlations = {}
    
    # Calculate additional XY-specific quantities for correlation analysis
    xy_model = create_xy_2d_model()
    helicity_moduli = []
    vorticities = []
    
    for config in configurations:
        helicity = calculate_helicity_modulus(config, xy_model)
        vorticity = xy_model.compute_vorticity(config)
        helicity_moduli.append(helicity)
        vorticities.append(vorticity)
    
    helicity_moduli = np.array(helicity_moduli)
    vorticities = np.array(vorticities)
    
    for dim in range(latent_representations.shape[1]):
        latent_dim = latent_representations[:, dim]
        
        # Correlation with temperature
        temp_corr = np.corrcoef(latent_dim, temperatures)[0, 1]
        
        # Correlation with order parameter (magnetization)
        order_corr = np.corrcoef(latent_dim, order_parameters)[0, 1]
        
        # Correlation with helicity modulus (KT order parameter)
        helicity_corr = np.corrcoef(latent_dim, helicity_moduli)[0, 1]
        
        # Correlation with vorticity
        vorticity_corr = np.corrcoef(latent_dim, vorticities)[0, 1]
        
        correlations[f'dim_{dim}'] = {
            'temperature_correlation': temp_corr,
            'order_parameter_correlation': order_corr,
            'helicity_correlation': helicity_corr,
            'vorticity_correlation': vorticity_corr
        }
        
        logger.info(f"Latent dim {dim}: T_corr={temp_corr:.3f}, M_corr={order_corr:.3f}, "
                   f"H_corr={helicity_corr:.3f}, V_corr={vorticity_corr:.3f}")
    
    # Find best latent dimension (highest correlation with helicity modulus for KT transition)
    best_dim = max(correlations.keys(), 
                  key=lambda k: abs(correlations[k]['helicity_correlation']))
    best_dim_idx = int(best_dim.split('_')[1])
    
    logger.info(f"Best latent dimension: {best_dim} "
               f"(helicity correlation: {correlations[best_dim]['helicity_correlation']:.3f})")
    
    return {
        'latent_representations': latent_representations,
        'correlations': correlations,
        'best_dimension': best_dim_idx,
        'best_latent_coords': latent_representations[:, best_dim_idx],
        'helicity_moduli': helicity_moduli,
        'vorticities': vorticities
    }


def create_xy_visualizations(data: Dict[str, Any],
                           kt_analysis: Dict[str, Any],
                           latent_analysis: Dict[str, Any],
                           output_dir: Path) -> None:
    """
    Create visualization plots for XY model analysis.
    
    Args:
        data: Generated XY model data
        kt_analysis: KT transition analysis results
        latent_analysis: Latent space analysis results
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. KT transition analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    temperatures = kt_analysis['temperatures']
    
    # Order parameter (magnetization)
    axes[0, 0].plot(temperatures, kt_analysis['mean_order_params'], 'bo-', markersize=4)
    axes[0, 0].axvline(kt_analysis['tc_theoretical'], color='r', linestyle='--', 
                      label=f'Theoretical Tc = {kt_analysis["tc_theoretical"]:.3f}')
    axes[0, 0].axvline(kt_analysis['tc_measured'], color='g', linestyle='--',
                      label=f'Measured Tc = {kt_analysis["tc_measured"]:.3f}')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Magnetization |M|')
    axes[0, 0].set_title('XY Model Magnetization')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Helicity modulus (KT order parameter)
    axes[0, 1].plot(temperatures, kt_analysis['mean_helicity_moduli'], 'ro-', markersize=4)
    axes[0, 1].axvline(kt_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[0, 1].axvline(kt_analysis['tc_measured'], color='g', linestyle='--')
    axes[0, 1].set_xlabel('Temperature')
    axes[0, 1].set_ylabel('Helicity Modulus')
    axes[0, 1].set_title('Superfluid Density (KT Order Parameter)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vorticity variance
    axes[0, 2].plot(temperatures, kt_analysis['vorticity_variances'], 'go-', markersize=4)
    axes[0, 2].axvline(kt_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[0, 2].axvline(kt_analysis['tc_measured'], color='g', linestyle='--')
    axes[0, 2].set_xlabel('Temperature')
    axes[0, 2].set_ylabel('Vorticity Variance')
    axes[0, 2].set_title('Vortex Unbinding')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Susceptibility
    axes[1, 0].plot(temperatures, kt_analysis['susceptibilities'], 'mo-', markersize=4)
    axes[1, 0].axvline(kt_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[1, 0].axvline(kt_analysis['tc_measured'], color='g', linestyle='--')
    axes[1, 0].set_xlabel('Temperature')
    axes[1, 0].set_ylabel('Susceptibility')
    axes[1, 0].set_title('Magnetic Susceptibility')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Specific heat
    axes[1, 1].plot(temperatures, kt_analysis['specific_heats'], 'co-', markersize=4)
    axes[1, 1].axvline(kt_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[1, 1].axvline(kt_analysis['tc_measured'], color='g', linestyle='--')
    axes[1, 1].set_xlabel('Temperature')
    axes[1, 1].set_ylabel('Specific Heat')
    axes[1, 1].set_title('Specific Heat')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Latent space correlation with helicity
    best_latent = latent_analysis['best_latent_coords']
    helicity_moduli = latent_analysis['helicity_moduli']
    axes[1, 2].scatter(data['temperatures'], best_latent, c=helicity_moduli, 
                      cmap='plasma', alpha=0.6, s=10)
    axes[1, 2].set_xlabel('Temperature')
    axes[1, 2].set_ylabel(f'Latent Dimension {latent_analysis["best_dimension"]}')
    axes[1, 2].set_title('Latent Space vs Temperature')
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('Helicity Modulus')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'xy_kt_transition_analysis.png')
    
    # 2. Latent space phase diagram
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if latent_analysis['latent_representations'].shape[1] >= 2:
        latent_0 = latent_analysis['latent_representations'][:, 0]
        latent_1 = latent_analysis['latent_representations'][:, 1]
        
        scatter = ax.scatter(latent_0, latent_1, c=data['temperatures'], 
                           cmap='coolwarm', alpha=0.7, s=15)
        ax.set_xlabel('Latent Dimension 0')
        ax.set_ylabel('Latent Dimension 1')
        ax.set_title('XY Model Phase Diagram in Latent Space')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature')
        ax.grid(True, alpha=0.3)
    
    save_figure(fig, output_dir / 'xy_latent_phase_diagram.png')
    
    # 3. Sample XY configurations with spin vectors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select representative temperatures
    temp_indices = [0, len(temperatures)//4, len(temperatures)//2, 
                   3*len(temperatures)//4, len(temperatures)-1]
    
    for i, temp_idx in enumerate(temp_indices[:6]):
        if i >= 6:
            break
        
        row, col = i // 3, i % 3
        temp = temperatures[temp_idx]
        
        # Find a configuration at this temperature
        temp_mask = data['temperatures'] == temp
        temp_configs = data['configurations'][temp_mask]
        
        if len(temp_configs) > 0:
            config = temp_configs[0]
            
            # Create vector field plot
            height, width = config.shape
            
            # Subsample for visualization
            step = max(1, height // 16)
            y_indices = np.arange(0, height, step)
            x_indices = np.arange(0, width, step)
            
            Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
            
            # Get angles at subsampled points
            angles_sub = config[::step, ::step]
            
            # Convert to vector components
            U = np.cos(angles_sub)
            V = np.sin(angles_sub)
            
            # Create quiver plot
            axes[row, col].quiver(X, Y, U, V, angles_sub, cmap='hsv', 
                                scale=1, scale_units='xy', angles='xy')
            axes[row, col].set_title(f'T = {temp:.3f}')
            axes[row, col].set_aspect('equal')
            axes[row, col].set_xlim(0, width-1)
            axes[row, col].set_ylim(0, height-1)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'xy_sample_configurations.png')
    
    logger.info(f"Saved XY visualization plots to {output_dir}")


def save_xy_data(data: Dict[str, Any],
                kt_analysis: Dict[str, Any],
                latent_analysis: Dict[str, Any],
                output_file: Path) -> None:
    """
    Save XY model data and analysis results to HDF5 file.
    
    Args:
        data: Generated XY model data
        kt_analysis: KT transition analysis results
        latent_analysis: Latent space analysis results
        output_file: Output HDF5 file path
    """
    logger = logging.getLogger(__name__)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Raw data
        f.create_dataset('configurations', data=data['configurations'])
        f.create_dataset('temperatures', data=data['temperatures'])
        f.create_dataset('order_parameters', data=data['order_parameters'])
        f.create_dataset('energies', data=data['energies'])
        
        # KT analysis
        kt_group = f.create_group('kt_analysis')
        kt_group.create_dataset('tc_measured', data=kt_analysis['tc_measured'])
        kt_group.create_dataset('tc_theoretical', data=kt_analysis['tc_theoretical'])
        kt_group.create_dataset('tc_error_percent', data=kt_analysis['tc_error_percent'])
        kt_group.create_dataset('unique_temperatures', data=kt_analysis['temperatures'])
        kt_group.create_dataset('mean_order_params', data=kt_analysis['mean_order_params'])
        kt_group.create_dataset('mean_helicity_moduli', data=kt_analysis['mean_helicity_moduli'])
        kt_group.create_dataset('vorticity_variances', data=kt_analysis['vorticity_variances'])
        kt_group.create_dataset('susceptibilities', data=kt_analysis['susceptibilities'])
        kt_group.create_dataset('specific_heats', data=kt_analysis['specific_heats'])
        
        # Latent analysis
        latent_group = f.create_group('latent_analysis')
        latent_group.create_dataset('latent_representations', data=latent_analysis['latent_representations'])
        latent_group.create_dataset('best_dimension', data=latent_analysis['best_dimension'])
        latent_group.create_dataset('best_latent_coords', data=latent_analysis['best_latent_coords'])
        latent_group.create_dataset('helicity_moduli', data=latent_analysis['helicity_moduli'])
        latent_group.create_dataset('vorticities', data=latent_analysis['vorticities'])
        
        # Correlations
        corr_group = latent_group.create_group('correlations')
        for dim_name, corr_data in latent_analysis['correlations'].items():
            dim_group = corr_group.create_group(dim_name)
            for corr_type, corr_value in corr_data.items():
                dim_group.create_dataset(corr_type, data=corr_value)
        
        # Metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['model_name'] = data['model_info']['model_name']
        meta_group.attrs['lattice_size'] = data['simulation_metadata']['lattice_size']
        meta_group.attrs['n_temperatures'] = data['simulation_metadata']['n_temperatures']
        meta_group.attrs['n_configs_per_temp'] = data['simulation_metadata']['n_configs_per_temp']
        meta_group.attrs['generation_timestamp'] = datetime.now().isoformat()
    
    logger.info(f"Saved XY data to {output_file}")


def main():
    """Main function for XY model analysis."""
    parser = argparse.ArgumentParser(description='Generate and analyze 2D XY model data')
    parser.add_argument('--lattice-size', type=int, nargs=2, default=[16, 16],
                       help='Lattice dimensions (height width)')
    parser.add_argument('--temp-range', type=float, nargs=2, default=[0.5, 1.5],
                       help='Temperature range (T_min T_max)')
    parser.add_argument('--n-temps', type=int, default=30,
                       help='Number of temperature points')
    parser.add_argument('--n-configs', type=int, default=200,
                       help='Configurations per temperature')
    parser.add_argument('--coupling', type=float, default=1.0,
                       help='Coupling strength J')
    parser.add_argument('--equilibration', type=int, default=50000,
                       help='Equilibration steps per temperature')
    parser.add_argument('--sampling-interval', type=int, default=100,
                       help='Sampling interval between configurations')
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='VAE latent dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='VAE training epochs')
    parser.add_argument('--output-dir', type=str, default='results/xy_analysis',
                       help='Output directory')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting 2D XY model analysis")
    logger.info(f"Parameters: lattice_size={args.lattice_size}, "
               f"temp_range={args.temp_range}, n_temps={args.n_temps}")
    
    try:
        # 1. Generate XY data
        logger.info("Step 1: Generating XY model data...")
        data = generate_xy_data(
            lattice_size=tuple(args.lattice_size),
            temperature_range=tuple(args.temp_range),
            n_temperatures=args.n_temps,
            n_configs_per_temp=args.n_configs,
            coupling_strength=args.coupling,
            equilibration_steps=args.equilibration,
            sampling_interval=args.sampling_interval
        )
        
        # 2. Analyze KT transition
        logger.info("Step 2: Analyzing Kosterlitz-Thouless transition...")
        kt_analysis = analyze_kt_transition(data)
        
        # 3. Train VAE
        logger.info("Step 3: Training VAE on XY configurations...")
        vae, training_results = train_xy_vae(
            data=data,
            latent_dim=args.latent_dim,
            epochs=args.epochs
        )
        
        # 4. Extract latent representations
        logger.info("Step 4: Extracting latent representations...")
        latent_analysis = extract_xy_latent_representations(vae, data)
        
        # 5. Create visualizations
        logger.info("Step 5: Creating visualization plots...")
        create_xy_visualizations(data, kt_analysis, latent_analysis, output_dir)
        
        # 6. Save results
        logger.info("Step 6: Saving results...")
        output_file = output_dir / f'xy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        save_xy_data(data, kt_analysis, latent_analysis, output_file)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("2D XY MODEL KOSTERLITZ-THOULESS ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Theoretical Tc: {kt_analysis['tc_theoretical']:.4f}")
        logger.info(f"Measured Tc: {kt_analysis['tc_measured']:.4f}")
        logger.info(f"Accuracy: {100 - kt_analysis['tc_error_percent']:.1f}%")
        logger.info(f"Best latent dimension: {latent_analysis['best_dimension']}")
        best_dim_key = f"dim_{latent_analysis['best_dimension']}"
        helicity_corr = latent_analysis['correlations'][best_dim_key]['helicity_correlation']
        logger.info(f"Helicity correlation: {helicity_corr:.3f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()