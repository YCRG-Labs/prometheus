#!/usr/bin/env python3
"""
Latent Space Analysis Script for Prometheus Project

This script performs post-training analysis of the VAE latent space to discover
order parameters and detect phase transitions in the Ising model.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ConvolutionalVAE
from src.analysis import LatentAnalyzer, OrderParameterDiscovery, PhaseDetector
from src.data import DataPreprocessor
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import ReproducibilityManager


def main():
    parser = argparse.ArgumentParser(description='Analyze VAE latent space for physics discovery')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained VAE model')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed HDF5 dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory for analysis results')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--batch-size', type=int, help='Batch size for encoding (overrides config)')
    parser.add_argument('--include-temperature-labels', action='store_true', 
                       help='Include temperature labels in analysis (for validation)')
    parser.add_argument('--save-latent-coords', action='store_true', 
                       help='Save latent coordinates to file')
    parser.add_argument('--analysis-only', action='store_true', 
                       help='Skip encoding and use existing latent coordinates')
    parser.add_argument('--latent-coords-file', type=str, 
                       help='Path to existing latent coordinates file (for --analysis-only)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Override configuration with command line arguments
    if args.output_dir:
        config.results_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Setup logging
    setup_logging(config.logging)
    
    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    
    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print("=" * 60)
    print("Prometheus Latent Space Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Device: {device}")
    print(f"  Output directory: {config.results_dir}")
    print(f"  Include temperature labels: {args.include_temperature_labels}")
    print(f"  Analysis only mode: {args.analysis_only}")
    if args.analysis_only and args.latent_coords_file:
        print(f"  Latent coordinates file: {args.latent_coords_file}")
    print()
    
    # Create output directory
    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.analysis_only:
        # Load and initialize model
        print("Loading trained VAE model...")
        model = ConvolutionalVAE(
            input_shape=tuple(config.vae.input_shape),
            latent_dim=config.vae.latent_dim,
            encoder_channels=config.vae.encoder_channels,
            decoder_channels=config.vae.decoder_channels,
            kernel_sizes=config.vae.kernel_sizes
        ).to(device)
        
        # Load model weights
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")
        
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_val_loss' in checkpoint:
                print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load dataset
        print("\nLoading dataset...")
        preprocessor = DataPreprocessor(config)
        
        # Verify dataset exists
        dataset_path = Path(args.data)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {args.data}")
        
        # Load dataset info
        dataset_info = preprocessor.load_dataset_info(args.data)
        print(f"  Total configurations: {dataset_info['n_configurations']}")
        print(f"  Configuration shape: {dataset_info['configuration_shape']}")
        
        # Create data loaders (use test set for analysis)
        _, _, test_loader = preprocessor.create_dataloaders(
            args.data,
            batch_size=config.training.batch_size,
            num_workers=4 if device.type == 'cuda' else 0
        )
        
        print(f"  Test batches: {len(test_loader)}")
        
        # Initialize latent analyzer
        analyzer = LatentAnalyzer(model, config, device)
        
        # Encode dataset to latent space
        print("\nEncoding dataset to latent space...")
        latent_coords, temperatures, magnetizations, energies = analyzer.encode_dataset(
            test_loader,
            include_temperature_labels=args.include_temperature_labels,
            include_physical_quantities=True
        )
        
        print(f"  Encoded {len(latent_coords)} configurations")
        print(f"  Latent space shape: {latent_coords.shape}")
        print(f"  Latent coordinate ranges:")
        for i in range(latent_coords.shape[1]):
            z_min, z_max = latent_coords[:, i].min(), latent_coords[:, i].max()
            z_mean, z_std = latent_coords[:, i].mean(), latent_coords[:, i].std()
            print(f"    z{i+1}: [{z_min:.3f}, {z_max:.3f}], mean={z_mean:.3f}, std={z_std:.3f}")
        
        # Save latent coordinates if requested
        if args.save_latent_coords:
            coords_file = output_dir / "latent_coordinates.npz"
            np.savez(
                coords_file,
                latent_coords=latent_coords,
                temperatures=temperatures,
                magnetizations=magnetizations,
                energies=energies
            )
            print(f"  Latent coordinates saved to: {coords_file}")
    
    else:
        # Load existing latent coordinates
        if not args.latent_coords_file:
            raise ValueError("--analysis-only requires --latent-coords-file")
        
        coords_file = Path(args.latent_coords_file)
        if not coords_file.exists():
            raise FileNotFoundError(f"Latent coordinates file not found: {args.latent_coords_file}")
        
        print(f"Loading existing latent coordinates from: {coords_file}")
        data = np.load(coords_file)
        latent_coords = data['latent_coords']
        temperatures = data['temperatures']
        magnetizations = data['magnetizations']
        energies = data['energies']
        
        print(f"  Loaded {len(latent_coords)} configurations")
        print(f"  Latent space shape: {latent_coords.shape}")
    
    # Order parameter discovery
    print("\nDiscovering order parameters...")
    order_param_discovery = OrderParameterDiscovery(config)
    
    order_param_results = order_param_discovery.discover_order_parameters(
        latent_coords=latent_coords,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=energies
    )
    
    print(f"  Order parameter analysis complete")
    print(f"  Primary order parameter dimension: z{order_param_results['primary_dimension'] + 1}")
    print(f"  Correlation with magnetization: {order_param_results['magnetization_correlation']:.4f}")
    print(f"  Statistical significance: p = {order_param_results['correlation_p_value']:.2e}")
    
    # Phase detection
    print("\nDetecting phase transitions...")
    phase_detector = PhaseDetector(config)
    
    phase_results = phase_detector.detect_phase_transition(
        latent_coords=latent_coords,
        temperatures=temperatures
    )
    
    print(f"  Phase detection complete")
    print(f"  Detected critical temperature: {phase_results['critical_temperature']:.4f}")
    print(f"  Theoretical critical temperature: {config.ising.critical_temp:.4f}")
    print(f"  Relative error: {abs(phase_results['critical_temperature'] - config.ising.critical_temp) / config.ising.critical_temp * 100:.2f}%")
    print(f"  Detection confidence: {phase_results['confidence']:.4f}")
    
    # Clustering analysis
    clustering_results = phase_detector.cluster_analysis(
        latent_coords=latent_coords,
        temperatures=temperatures,
        n_clusters=2
    )
    
    print(f"  Clustering analysis:")
    print(f"    Silhouette score: {clustering_results['silhouette_score']:.4f}")
    print(f"    Cluster separation: {clustering_results['cluster_separation']:.4f}")
    
    # Save analysis results
    print("\nSaving analysis results...")
    
    # Combine all results
    analysis_results = {
        'order_parameters': order_param_results,
        'phase_detection': phase_results,
        'clustering': clustering_results,
        'configuration': {
            'model_path': str(args.model) if not args.analysis_only else None,
            'data_path': str(args.data),
            'latent_dim': config.vae.latent_dim,
            'n_configurations': len(latent_coords),
            'theoretical_critical_temp': config.ising.critical_temp
        }
    }
    
    # Save results as JSON
    results_file = output_dir / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"  Analysis results saved to: {results_file}")
    
    # Save detailed data
    detailed_file = output_dir / "detailed_analysis.npz"
    np.savez(
        detailed_file,
        latent_coords=latent_coords,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=energies,
        cluster_labels=clustering_results['labels'],
        order_parameter_values=order_param_results['order_parameter_values']
    )
    print(f"  Detailed analysis data saved to: {detailed_file}")
    
    print("\n" + "=" * 60)
    print("Latent Space Analysis Complete!")
    print("=" * 60)
    print(f"Key Findings:")
    print(f"  • Discovered order parameter in latent dimension z{order_param_results['primary_dimension'] + 1}")
    print(f"  • Magnetization correlation: {order_param_results['magnetization_correlation']:.4f}")
    print(f"  • Critical temperature: {phase_results['critical_temperature']:.4f} (theory: {config.ising.critical_temp:.4f})")
    print(f"  • Relative error: {abs(phase_results['critical_temperature'] - config.ising.critical_temp) / config.ising.critical_temp * 100:.2f}%")
    print(f"  • Phase separation quality: {clustering_results['silhouette_score']:.4f}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Ready for visualization and physics validation.")


if __name__ == "__main__":
    main()