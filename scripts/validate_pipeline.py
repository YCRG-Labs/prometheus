#!/usr/bin/env python3
"""
Pipeline Configuration Validation Utility

This utility validates the Prometheus pipeline configuration and checks
for common issues before running the full pipeline.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PrometheusConfig, ConfigLoader


def validate_config_file(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a configuration file."""
    errors = []
    
    if not config_path.exists():
        errors.append(f"Configuration file not found: {config_path}")
        return False, errors
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Could not read configuration file: {e}")
        return False, errors
    
    # Check required sections
    required_sections = ['ising', 'vae', 'training', 'logging']
    for section in required_sections:
        if section not in config_data:
            errors.append(f"Missing required section: {section}")
    
    # Validate ising section
    if 'ising' in config_data:
        ising = config_data['ising']
        
        if 'lattice_size' not in ising:
            errors.append("Missing ising.lattice_size")
        elif not isinstance(ising['lattice_size'], list) or len(ising['lattice_size']) != 2:
            errors.append("ising.lattice_size must be a list of 2 integers")
        elif any(not isinstance(x, int) or x <= 0 for x in ising['lattice_size']):
            errors.append("ising.lattice_size values must be positive integers")
        
        if 'temperature_range' not in ising:
            errors.append("Missing ising.temperature_range")
        elif not isinstance(ising['temperature_range'], list) or len(ising['temperature_range']) != 2:
            errors.append("ising.temperature_range must be a list of 2 numbers")
        elif ising['temperature_range'][0] >= ising['temperature_range'][1]:
            errors.append("ising.temperature_range[0] must be less than temperature_range[1]")
        
        required_ising_params = ['n_temperatures', 'n_configs_per_temp', 'equilibration_steps']
        for param in required_ising_params:
            if param not in ising:
                errors.append(f"Missing ising.{param}")
            elif not isinstance(ising[param], int) or ising[param] <= 0:
                errors.append(f"ising.{param} must be a positive integer")
    
    # Validate VAE section
    if 'vae' in config_data:
        vae = config_data['vae']
        
        if 'latent_dim' not in vae:
            errors.append("Missing vae.latent_dim")
        elif not isinstance(vae['latent_dim'], int) or vae['latent_dim'] <= 0:
            errors.append("vae.latent_dim must be a positive integer")
        
        if 'input_shape' not in vae:
            errors.append("Missing vae.input_shape")
        elif not isinstance(vae['input_shape'], list) or len(vae['input_shape']) != 3:
            errors.append("vae.input_shape must be a list of 3 integers")
    
    # Validate training section
    if 'training' in config_data:
        training = config_data['training']
        
        required_training_params = ['batch_size', 'num_epochs']
        for param in required_training_params:
            if param not in training:
                errors.append(f"Missing training.{param}")
            elif not isinstance(training[param], int) or training[param] <= 0:
                errors.append(f"training.{param} must be a positive integer")
        
        if 'learning_rate' not in training:
            errors.append("Missing training.learning_rate")
        elif not isinstance(training['learning_rate'], (int, float)) or training['learning_rate'] <= 0:
            errors.append("training.learning_rate must be a positive number")
    
    return len(errors) == 0, errors


def check_dependencies() -> tuple[bool, list[str]]:
    """Check for required dependencies and system requirements."""
    errors = []
    
    # Check Python packages
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'scipy', 'scikit-learn', 
        'h5py', 'yaml', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Missing required package: {package}")
    
    # Check for CUDA availability (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("ℹ CUDA not available, will use CPU")
    except ImportError:
        pass
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description='Validate Prometheus pipeline configuration')
    parser.add_argument('--config', type=str, help='Path to configuration file to validate')
    parser.add_argument('--check-deps', action='store_true', help='Check system dependencies')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Prometheus Pipeline Configuration Validator")
    print("=" * 60)
    
    all_valid = True
    
    # Check dependencies if requested
    if args.check_deps:
        print("\nChecking system dependencies...")
        deps_valid, dep_errors = check_dependencies()
        
        if deps_valid:
            print("✓ All required dependencies are available")
        else:
            print("✗ Missing dependencies:")
            for error in dep_errors:
                print(f"  - {error}")
            all_valid = False
    
    # Validate configuration
    if args.config:
        config_path = Path(args.config)
        print(f"\nValidating configuration file: {config_path}")
        
        config_valid, config_errors = validate_config_file(config_path)
        
        if config_valid:
            print("✓ Configuration file is valid")
            
            # Try loading with ConfigLoader
            try:
                config_loader = ConfigLoader()
                config = config_loader.load_config(str(config_path))
                print("✓ Configuration loads successfully")
                
                if args.verbose:
                    print(f"\nConfiguration summary:")
                    print(f"  Lattice size: {config.ising.lattice_size}")
                    print(f"  Temperature range: {config.ising.temperature_range}")
                    print(f"  Total configurations: {config.ising.n_temperatures * config.ising.n_configs_per_temp:,}")
                    print(f"  VAE latent dimensions: {config.vae.latent_dim}")
                    print(f"  Training epochs: {config.training.num_epochs}")
                    print(f"  Batch size: {config.training.batch_size}")
                
            except Exception as e:
                print(f"✗ Configuration loading failed: {e}")
                all_valid = False
        else:
            print("✗ Configuration file has errors:")
            for error in config_errors:
                print(f"  - {error}")
            all_valid = False
    
    else:
        # Validate default configuration
        print("\nValidating default configuration...")
        try:
            config = PrometheusConfig()
            print("✓ Default configuration is valid")
            
            if args.verbose:
                print(f"\nDefault configuration summary:")
                print(f"  Lattice size: {config.ising.lattice_size}")
                print(f"  Temperature range: {config.ising.temperature_range}")
                print(f"  Total configurations: {config.ising.n_temperatures * config.ising.n_configs_per_temp:,}")
                print(f"  VAE latent dimensions: {config.vae.latent_dim}")
                print(f"  Training epochs: {config.training.num_epochs}")
                print(f"  Batch size: {config.training.batch_size}")
        
        except Exception as e:
            print(f"✗ Default configuration failed: {e}")
            all_valid = False
    
    # Check script files
    print("\nChecking pipeline scripts...")
    scripts_dir = Path(__file__).parent
    required_scripts = [
        'generate_data.py',
        'train_vae.py', 
        'analyze_latent_space.py',
        'generate_visualizations.py',
        'run_prometheus_pipeline.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (missing)")
            missing_scripts.append(script)
            all_valid = False
    
    if missing_scripts:
        print(f"\nMissing {len(missing_scripts)} required scripts")
    else:
        print("✓ All required scripts are present")
    
    # Final result
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ VALIDATION PASSED")
        print("Pipeline is ready to run!")
    else:
        print("✗ VALIDATION FAILED")
        print("Please fix the issues above before running the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()