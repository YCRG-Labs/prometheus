#!/usr/bin/env python3
"""
Generate Q=3 Potts Model Dataset and Apply Prometheus Analysis.

This script generates a comprehensive dataset for the Q=3 Potts model,
applies the Prometheus VAE to learn representations, and detects the
first-order phase transition at Tc ≈ 1.005/J.
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

from data.unified_monte_carlo import create_potts_simulator
from models.physics_models import create_potts_3state_model
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
            logging.FileHandler(f'logs/potts_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_potts_data(lattice_size: Tuple[int, int],
                       temperature_range: Tuple[float, float],
                       n_temperatures: int = 50,
                       n_configs_per_temp: int = 200,
                       coupling_strength: float = 1.0,
                       equilibration_steps: int = 50000,
                       sampling_interval: int = 100) -> Dict[str, Any]:
    """
    Generate Q=3 Potts model dataset.
    
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
    
    logger.info(f"Generating Q=3 Potts data: lattice_size={lattice_size}, "
               f"T_range={temperature_range}, n_temps={n_temperatures}")
    
    # Create Potts simulator
    simulator = create_potts_simulator(
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
    
    # Calculate additional properties
    logger.info("Computing additional thermodynamic properties...")
    
    # Group data by temperature for analysis
    temperatures = np.unique(result.temperatures)
    temp_data = {}
    
    for temp in temperatures:
        mask = result.temperatures == temp
        temp_configs = result.configurations[mask]
        temp_order_params = result.order_parameters[mask]
        temp_energies = result.energies[mask]
        
        # Calculate statistics
        mean_order_param = np.mean(temp_order_params)
        std_order_param = np.std(temp_order_params)
        mean_energy = np.mean(temp_energies)
        std_energy = np.std(temp_energies)
        
        # Calculate susceptibility (χ = N * <M²> - <M>²) / T)
        susceptibility = len(temp_order_params) * (np.mean(temp_order_params**2) - mean_order_param**2) / temp
        
        # Calculate specific heat (C = (<E²> - <E>²) / T²)
        specific_heat = (np.mean(temp_energies**2) - mean_energy**2) / (temp**2)
        
        temp_data[temp] = {
            'configurations': temp_configs,
            'order_parameters': temp_order_params,
            'energies': temp_energies,
            'mean_order_param': mean_order_param,
            'std_order_param': std_order_param,
            'mean_energy': mean_energy,
            'std_energy': std_energy,
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


def analyze_phase_transition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze phase transition properties of Potts model data.
    
    Args:
        data: Generated Potts model data
        
    Returns:
        Dictionary with phase transition analysis results
    """
    logger = logging.getLogger(__name__)
    
    temperatures = data['unique_temperatures']
    temp_data = data['temperature_data']
    
    # Extract thermodynamic quantities
    mean_order_params = [temp_data[T]['mean_order_param'] for T in temperatures]
    susceptibilities = [temp_data[T]['susceptibility'] for T in temperatures]
    specific_heats = [temp_data[T]['specific_heat'] for T in temperatures]
    
    # Find critical temperature from susceptibility peak
    max_susceptibility_idx = np.argmax(susceptibilities)
    tc_susceptibility = temperatures[max_susceptibility_idx]
    
    # Find critical temperature from specific heat peak
    max_specific_heat_idx = np.argmax(specific_heats)
    tc_specific_heat = temperatures[max_specific_heat_idx]
    
    # Theoretical critical temperature for Q=3 Potts
    tc_theoretical = data['model_info']['theoretical_tc']
    
    logger.info(f"Critical temperature analysis:")
    logger.info(f"  From susceptibility peak: Tc = {tc_susceptibility:.4f}")
    logger.info(f"  From specific heat peak: Tc = {tc_specific_heat:.4f}")
    logger.info(f"  Theoretical value: Tc = {tc_theoretical:.4f}")
    
    # Calculate accuracy
    tc_measured = tc_susceptibility  # Use susceptibility peak as primary estimate
    tc_error = abs(tc_measured - tc_theoretical) / tc_theoretical * 100
    
    logger.info(f"  Critical temperature accuracy: {100 - tc_error:.1f}% "
               f"(error: {tc_error:.1f}%)")
    
    return {
        'tc_susceptibility': tc_susceptibility,
        'tc_specific_heat': tc_specific_heat,
        'tc_theoretical': tc_theoretical,
        'tc_measured': tc_measured,
        'tc_error_percent': tc_error,
        'temperatures': temperatures,
        'mean_order_params': np.array(mean_order_params),
        'susceptibilities': np.array(susceptibilities),
        'specific_heats': np.array(specific_heats)
    }


def train_potts_vae(data: Dict[str, Any], 
                   latent_dim: int = 2,
                   epochs: int = 100,
                   batch_size: int = 64) -> Tuple[AdaptiveVAE, Dict[str, Any]]:
    """
    Train VAE on Potts model configurations.
    
    Args:
        data: Generated Potts model data
        latent_dim: Latent space dimensionality
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Tuple of (trained_vae, training_results)
    """
    logger = logging.getLogger(__name__)
    
    configurations = data['configurations']
    temperatures = data['temperatures']
    
    logger.info(f"Training VAE on {len(configurations)} Potts configurations")
    logger.info(f"Configuration shape: {configurations[0].shape}")
    
    # Prepare data for VAE (normalize to [0, 1] for Potts states {0, 1, 2})
    # Convert to one-hot encoding for better VAE training
    n_configs, height, width = configurations.shape
    n_states = 3  # Q=3 Potts
    
    # One-hot encode configurations
    configs_onehot = np.zeros((n_configs, height, width, n_states))
    for i, config in enumerate(configurations):
        for state in range(n_states):
            configs_onehot[i, :, :, state] = (config == state).astype(float)
    
    # Reshape for VAE input: (batch, channels, height, width)
    vae_input = configs_onehot.transpose(0, 3, 1, 2)
    
    logger.info(f"VAE input shape: {vae_input.shape}")
    
    # Create adaptive VAE
    vae = AdaptiveVAE(
        input_shape=(n_states, height, width),
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


def extract_latent_representations(vae: AdaptiveVAE, 
                                 data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract latent representations from trained VAE.
    
    Args:
        vae: Trained VAE model
        data: Potts model data
        
    Returns:
        Dictionary with latent representations and analysis
    """
    logger = logging.getLogger(__name__)
    
    configurations = data['configurations']
    temperatures = data['temperatures']
    order_parameters = data['order_parameters']
    
    # Prepare configurations (same as training)
    n_configs, height, width = configurations.shape
    n_states = 3
    
    configs_onehot = np.zeros((n_configs, height, width, n_states))
    for i, config in enumerate(configurations):
        for state in range(n_states):
            configs_onehot[i, :, :, state] = (config == state).astype(float)
    
    vae_input = configs_onehot.transpose(0, 3, 1, 2)
    
    logger.info("Extracting latent representations...")
    
    # Extract latent representations
    latent_representations = vae.encode(vae_input)
    
    logger.info(f"Extracted latent representations: shape {latent_representations.shape}")
    
    # Analyze correlations with physical quantities
    correlations = {}
    
    for dim in range(latent_representations.shape[1]):
        latent_dim = latent_representations[:, dim]
        
        # Correlation with temperature
        temp_corr = np.corrcoef(latent_dim, temperatures)[0, 1]
        
        # Correlation with order parameter
        order_corr = np.corrcoef(latent_dim, order_parameters)[0, 1]
        
        correlations[f'dim_{dim}'] = {
            'temperature_correlation': temp_corr,
            'order_parameter_correlation': order_corr
        }
        
        logger.info(f"Latent dim {dim}: T_corr={temp_corr:.3f}, M_corr={order_corr:.3f}")
    
    # Find best latent dimension (highest correlation with order parameter)
    best_dim = max(correlations.keys(), 
                  key=lambda k: abs(correlations[k]['order_parameter_correlation']))
    best_dim_idx = int(best_dim.split('_')[1])
    
    logger.info(f"Best latent dimension: {best_dim} "
               f"(order param correlation: {correlations[best_dim]['order_parameter_correlation']:.3f})")
    
    return {
        'latent_representations': latent_representations,
        'correlations': correlations,
        'best_dimension': best_dim_idx,
        'best_latent_coords': latent_representations[:, best_dim_idx]
    }


def create_potts_visualizations(data: Dict[str, Any],
                               phase_analysis: Dict[str, Any],
                               latent_analysis: Dict[str, Any],
                               output_dir: Path) -> None:
    """
    Create visualization plots for Potts model analysis.
    
    Args:
        data: Generated Potts model data
        phase_analysis: Phase transition analysis results
        latent_analysis: Latent space analysis results
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Thermodynamic quantities vs temperature
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    temperatures = phase_analysis['temperatures']
    
    # Order parameter
    axes[0, 0].plot(temperatures, phase_analysis['mean_order_params'], 'bo-', markersize=4)
    axes[0, 0].axvline(phase_analysis['tc_theoretical'], color='r', linestyle='--', 
                      label=f'Theoretical Tc = {phase_analysis["tc_theoretical"]:.3f}')
    axes[0, 0].axvline(phase_analysis['tc_measured'], color='g', linestyle='--',
                      label=f'Measured Tc = {phase_analysis["tc_measured"]:.3f}')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Order Parameter')
    axes[0, 0].set_title('Q=3 Potts Order Parameter')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Susceptibility
    axes[0, 1].plot(temperatures, phase_analysis['susceptibilities'], 'ro-', markersize=4)
    axes[0, 1].axvline(phase_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[0, 1].axvline(phase_analysis['tc_measured'], color='g', linestyle='--')
    axes[0, 1].set_xlabel('Temperature')
    axes[0, 1].set_ylabel('Susceptibility')
    axes[0, 1].set_title('Susceptibility (First-Order Peak)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Specific heat
    axes[1, 0].plot(temperatures, phase_analysis['specific_heats'], 'go-', markersize=4)
    axes[1, 0].axvline(phase_analysis['tc_theoretical'], color='r', linestyle='--')
    axes[1, 0].axvline(phase_analysis['tc_measured'], color='g', linestyle='--')
    axes[1, 0].set_xlabel('Temperature')
    axes[1, 0].set_ylabel('Specific Heat')
    axes[1, 0].set_title('Specific Heat')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent space correlation
    best_latent = latent_analysis['best_latent_coords']
    axes[1, 1].scatter(data['temperatures'], best_latent, c=data['order_parameters'], 
                      cmap='viridis', alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Temperature')
    axes[1, 1].set_ylabel(f'Latent Dimension {latent_analysis["best_dimension"]}')
    axes[1, 1].set_title('Latent Space vs Temperature')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Order Parameter')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'potts_thermodynamic_analysis.png')
    
    # 2. Latent space phase diagram
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if latent_analysis['latent_representations'].shape[1] >= 2:
        latent_0 = latent_analysis['latent_representations'][:, 0]
        latent_1 = latent_analysis['latent_representations'][:, 1]
        
        scatter = ax.scatter(latent_0, latent_1, c=data['temperatures'], 
                           cmap='coolwarm', alpha=0.7, s=15)
        ax.set_xlabel('Latent Dimension 0')
        ax.set_ylabel('Latent Dimension 1')
        ax.set_title('Q=3 Potts Phase Diagram in Latent Space')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature')
        ax.grid(True, alpha=0.3)
    
    save_figure(fig, output_dir / 'potts_latent_phase_diagram.png')
    
    # 3. Sample configurations at different temperatures
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
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
            
            # Create color map for Potts states
            im = axes[row, col].imshow(config, cmap='Set1', vmin=0, vmax=2)
            axes[row, col].set_title(f'T = {temp:.3f}')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'potts_sample_configurations.png')
    
    logger.info(f"Saved Potts visualization plots to {output_dir}")


def save_potts_data(data: Dict[str, Any],
                   phase_analysis: Dict[str, Any],
                   latent_analysis: Dict[str, Any],
                   output_file: Path) -> None:
    """
    Save Potts model data and analysis results to HDF5 file.
    
    Args:
        data: Generated Potts model data
        phase_analysis: Phase transition analysis results
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
        
        # Phase analysis
        phase_group = f.create_group('phase_analysis')
        phase_group.create_dataset('tc_measured', data=phase_analysis['tc_measured'])
        phase_group.create_dataset('tc_theoretical', data=phase_analysis['tc_theoretical'])
        phase_group.create_dataset('tc_error_percent', data=phase_analysis['tc_error_percent'])
        phase_group.create_dataset('unique_temperatures', data=phase_analysis['temperatures'])
        phase_group.create_dataset('mean_order_params', data=phase_analysis['mean_order_params'])
        phase_group.create_dataset('susceptibilities', data=phase_analysis['susceptibilities'])
        phase_group.create_dataset('specific_heats', data=phase_analysis['specific_heats'])
        
        # Latent analysis
        latent_group = f.create_group('latent_analysis')
        latent_group.create_dataset('latent_representations', data=latent_analysis['latent_representations'])
        latent_group.create_dataset('best_dimension', data=latent_analysis['best_dimension'])
        latent_group.create_dataset('best_latent_coords', data=latent_analysis['best_latent_coords'])
        
        # Correlations
        corr_group = latent_group.create_group('correlations')
        for dim_name, corr_data in latent_analysis['correlations'].items():
            dim_group = corr_group.create_group(dim_name)
            dim_group.create_dataset('temperature_correlation', data=corr_data['temperature_correlation'])
            dim_group.create_dataset('order_parameter_correlation', data=corr_data['order_parameter_correlation'])
        
        # Metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['model_name'] = data['model_info']['model_name']
        meta_group.attrs['lattice_size'] = data['simulation_metadata']['lattice_size']
        meta_group.attrs['n_temperatures'] = data['simulation_metadata']['n_temperatures']
        meta_group.attrs['n_configs_per_temp'] = data['simulation_metadata']['n_configs_per_temp']
        meta_group.attrs['generation_timestamp'] = datetime.now().isoformat()
    
    logger.info(f"Saved Potts data to {output_file}")


def main():
    """Main function for Potts model analysis."""
    parser = argparse.ArgumentParser(description='Generate and analyze Q=3 Potts model data')
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
    parser.add_argument('--output-dir', type=str, default='results/potts_analysis',
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
    
    logger.info("Starting Q=3 Potts model analysis")
    logger.info(f"Parameters: lattice_size={args.lattice_size}, "
               f"temp_range={args.temp_range}, n_temps={args.n_temps}")
    
    try:
        # 1. Generate Potts data
        logger.info("Step 1: Generating Potts model data...")
        data = generate_potts_data(
            lattice_size=tuple(args.lattice_size),
            temperature_range=tuple(args.temp_range),
            n_temperatures=args.n_temps,
            n_configs_per_temp=args.n_configs,
            coupling_strength=args.coupling,
            equilibration_steps=args.equilibration,
            sampling_interval=args.sampling_interval
        )
        
        # 2. Analyze phase transition
        logger.info("Step 2: Analyzing phase transition...")
        phase_analysis = analyze_phase_transition(data)
        
        # 3. Train VAE
        logger.info("Step 3: Training VAE on Potts configurations...")
        vae, training_results = train_potts_vae(
            data=data,
            latent_dim=args.latent_dim,
            epochs=args.epochs
        )
        
        # 4. Extract latent representations
        logger.info("Step 4: Extracting latent representations...")
        latent_analysis = extract_latent_representations(vae, data)
        
        # 5. Create visualizations
        logger.info("Step 5: Creating visualization plots...")
        create_potts_visualizations(data, phase_analysis, latent_analysis, output_dir)
        
        # 6. Save results
        logger.info("Step 6: Saving results...")
        output_file = output_dir / f'potts_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        save_potts_data(data, phase_analysis, latent_analysis, output_file)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Q=3 POTTS MODEL ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Theoretical Tc: {phase_analysis['tc_theoretical']:.4f}")
        logger.info(f"Measured Tc: {phase_analysis['tc_measured']:.4f}")
        logger.info(f"Accuracy: {100 - phase_analysis['tc_error_percent']:.1f}%")
        logger.info(f"Best latent dimension: {latent_analysis['best_dimension']}")
        logger.info(f"Order parameter correlation: {latent_analysis['correlations'][f'dim_{latent_analysis['best_dimension']}']['order_parameter_correlation']:.3f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()