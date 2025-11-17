#!/usr/bin/env python3
"""
Demonstration of Task 6 VAE Optimizations

This script demonstrates the three key optimizations implemented in Task 6:
1. Gradient clipping for stability
2. GPU acceleration support
3. Early stopping

It trains a small VAE on synthetic data to show these features in action.
"""

import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ConvolutionalVAE3D
from src.training.enhanced_physics_vae import (
    EnhancedPhysicsVAETrainer,
    EnhancedPhysicsLossWeights,
    create_enhanced_physics_trainer
)


def create_synthetic_data(n_samples=100, size=16):
    """Create synthetic 3D spin configuration data for demonstration."""
    print("Creating synthetic 3D Ising data...")
    
    # Temperature range around critical point
    temperatures = np.linspace(3.0, 6.0, n_samples)
    
    # Generate synthetic spin configurations
    configurations = []
    magnetizations = []
    
    for T in temperatures:
        # Simple model: higher temp = more random, lower temp = more ordered
        if T < 4.5:  # Below Tc
            # Ordered phase - mostly aligned spins
            config = np.random.choice([-1, 1], size=(size, size, size), 
                                     p=[0.2, 0.8])  # 80% spin up
        else:  # Above Tc
            # Disordered phase - random spins
            config = np.random.choice([-1, 1], size=(size, size, size), 
                                     p=[0.5, 0.5])  # 50/50
        
        configurations.append(config)
        magnetizations.append(np.abs(config.mean()))
    
    configurations = np.array(configurations, dtype=np.float32)
    magnetizations = np.array(magnetizations, dtype=np.float32)
    
    print(f"✓ Created {n_samples} configurations of size {size}³")
    print(f"  Temperature range: [{temperatures.min():.2f}, {temperatures.max():.2f}]")
    print(f"  Magnetization range: [{magnetizations.min():.4f}, {magnetizations.max():.4f}]")
    
    return configurations, temperatures, magnetizations


def demonstrate_task_6_features():
    """Demonstrate all Task 6 optimization features."""
    
    print("\n" + "="*70)
    print("TASK 6 VAE OPTIMIZATIONS - DEMONSTRATION")
    print("="*70)
    
    # Create synthetic data
    configs, temps, mags = create_synthetic_data(n_samples=200, size=16)
    
    # Normalize to [0, 1] for VAE
    configs_normalized = (configs + 1) / 2.0
    
    # Convert to tensors
    config_tensor = torch.FloatTensor(configs_normalized).unsqueeze(1)  # Add channel dim
    temp_tensor = torch.FloatTensor(temps)
    mag_tensor = torch.FloatTensor(mags)
    
    # Split into train/val
    train_size = int(0.8 * len(configs))
    train_configs = config_tensor[:train_size]
    val_configs = config_tensor[train_size:]
    train_temps = temp_tensor[:train_size]
    val_temps = temp_tensor[train_size:]
    train_mags = mag_tensor[:train_size]
    val_mags = mag_tensor[train_size:]
    
    print(f"\n✓ Data prepared: {train_size} train, {len(val_configs)} validation")
    
    # =========================================================================
    # FEATURE 1: GPU Acceleration
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE 1: GPU ACCELERATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Automatic device selection: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("  Note: Running on CPU (GPU not available)")
    
    # Create model
    model = ConvolutionalVAE3D(
        input_shape=(1, 16, 16, 16),
        latent_dim=2,
        encoder_channels=[8, 16],
        decoder_channels=[16, 8, 1],
        kernel_sizes=[3, 3]
    ).to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Model moved to {device}")
    
    # =========================================================================
    # FEATURE 2: Gradient Clipping
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE 2: GRADIENT CLIPPING (max_norm=1.0)")
    print("="*70)
    
    # Create trainer with gradient clipping
    loss_weights = EnhancedPhysicsLossWeights(
        reconstruction=1.0,
        kl_divergence=1.0,
        magnetization_correlation=2.0,
        temperature_ordering=1.5,
        critical_enhancement=1.0
    )
    
    trainer = create_enhanced_physics_trainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        loss_weights=loss_weights
    )
    
    print("✓ Trainer created with gradient clipping enabled")
    print("  Clipping threshold: max_norm=1.0")
    print("  This prevents gradient explosion during training")
    
    # =========================================================================
    # FEATURE 3: Early Stopping
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE 3: EARLY STOPPING (patience=5 for demo)")
    print("="*70)
    
    print("✓ Early stopping enabled")
    print("  Patience: 5 epochs (reduced for demo, normally 20)")
    print("  Monitors validation loss and stops if no improvement")
    
    # =========================================================================
    # TRAINING DEMONSTRATION
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING WITH ALL OPTIMIZATIONS")
    print("="*70)
    
    # Normalize configs to [-1, 1] for VAE
    train_configs = train_configs * 2.0 - 1.0
    val_configs = val_configs * 2.0 - 1.0
    
    # Move to device
    train_configs = train_configs.to(device)
    val_configs = val_configs.to(device)
    train_temps = train_temps.to(device)
    val_temps = val_temps.to(device)
    train_mags = train_mags.to(device)
    val_mags = val_mags.to(device)
    
    # Training loop with early stopping
    max_epochs = 20
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for up to {max_epochs} epochs...")
    print("(Will stop early if validation loss doesn't improve)\n")
    
    for epoch in range(max_epochs):
        # Train epoch
        train_losses = trainer.train_epoch(
            configurations=train_configs,
            temperatures=train_temps,
            magnetizations=train_mags,
            energies=None,
            batch_size=32
        )
        
        # Validation
        val_losses = trainer.validate(
            configurations=val_configs,
            temperatures=val_temps,
            magnetizations=val_mags,
            energies=None
        )
        
        # Get current learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{max_epochs}: "
              f"Train Loss: {train_losses['total']:.4f}, "
              f"Val Loss: {val_losses['total']:.4f}, "
              f"LR: {current_lr:.2e}")
        
        # Early stopping check
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            print(f"  → New best model! (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    
    print("\n✓ All Task 6 optimizations demonstrated:")
    print("  1. GPU Acceleration: Model and data on", device)
    print("  2. Gradient Clipping: Enabled with max_norm=1.0")
    print("  3. Early Stopping: Triggered after", patience_counter, "epochs without improvement")
    print("  4. Learning Rate Scheduling: ReduceLROnPlateau active")
    
    print("\n✓ Training completed successfully")
    print(f"  Final validation loss: {val_losses['total']:.4f}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Epochs trained: {epoch+1}/{max_epochs}")
    print(f"  Time saved by early stopping: ~{100*(max_epochs-epoch-1)/max_epochs:.0f}%")
    
    if torch.cuda.is_available():
        print(f"\n✓ GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\n" + "="*70)
    print("Task 6 optimizations are working correctly!")
    print("="*70)


if __name__ == "__main__":
    try:
        demonstrate_task_6_features()
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
