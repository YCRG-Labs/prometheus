#!/usr/bin/env python3
"""
Demonstration of Enhanced Training Features

This script demonstrates the enhanced training capabilities without
requiring the full project setup.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_advanced_schedulers():
    """Demonstrate advanced learning rate schedulers."""
    logger.info("=== Advanced Learning Rate Schedulers Demo ===")
    
    from training.advanced_schedulers import (
        CosineAnnealingWarmRestarts, WarmupCosineScheduler, 
        CyclicLRScheduler, create_advanced_scheduler
    )
    
    # Create a dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test Cosine Annealing with Warm Restarts
    logger.info("Testing Cosine Annealing with Warm Restarts...")
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    lrs = []
    for epoch in range(30):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    logger.info(f"LR range: {min(lrs):.2e} to {max(lrs):.2e}")
    logger.info(f"LR at epochs 0, 10, 20: {lrs[0]:.2e}, {lrs[10]:.2e}, {lrs[20]:.2e}")
    
    # Test Warmup Cosine Scheduler
    logger.info("Testing Warmup Cosine Scheduler...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Reset
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=20, eta_min=1e-6)
    
    warmup_lrs = []
    for epoch in range(20):
        warmup_lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    logger.info(f"Warmup LR progression: {warmup_lrs[:6]}")
    logger.info(f"Final LR: {warmup_lrs[-1]:.2e}")
    
    logger.info("Advanced schedulers demo completed!\n")


def demo_data_augmentation():
    """Demonstrate data augmentation techniques."""
    logger.info("=== Data Augmentation Demo ===")
    
    from training.data_augmentation import (
        RotationAugmentation, ReflectionAugmentation, SpinFlipAugmentation,
        create_standard_augmentation, test_augmentation_physics_preservation
    )
    
    # Create test spin configuration
    test_config = torch.randint(0, 2, (1, 8, 8)).float() * 2 - 1  # Convert to {-1, +1}
    logger.info(f"Original config shape: {test_config.shape}")
    logger.info(f"Original magnetization: {torch.mean(test_config).item():.3f}")
    
    # Test rotation augmentation
    logger.info("Testing rotation augmentation...")
    rotation_aug = RotationAugmentation(angles=[90, 180, 270], probability=1.0)
    rotated = rotation_aug.transform(test_config.clone())
    logger.info(f"Rotated config magnetization: {torch.mean(rotated).item():.3f}")
    
    # Test reflection augmentation
    logger.info("Testing reflection augmentation...")
    reflection_aug = ReflectionAugmentation(probability=1.0)
    reflected = reflection_aug.transform(test_config.clone())
    logger.info(f"Reflected config magnetization: {torch.mean(reflected).item():.3f}")
    
    # Test spin flip augmentation
    logger.info("Testing spin flip augmentation...")
    spin_flip_aug = SpinFlipAugmentation(probability=1.0)
    flipped = spin_flip_aug.transform(test_config.clone())
    logger.info(f"Flipped config magnetization: {torch.mean(flipped).item():.3f}")
    
    # Test physics preservation
    logger.info("Testing physics preservation...")
    standard_aug = create_standard_augmentation()
    stats = test_augmentation_physics_preservation(standard_aug, test_config, n_tests=10)
    logger.info(f"Mean energy difference: {stats['mean_energy_diff']:.6f}")
    logger.info(f"Mean magnetization difference: {stats['mean_mag_diff']:.6f}")
    
    logger.info("Data augmentation demo completed!\n")


def demo_progressive_training():
    """Demonstrate progressive training concepts."""
    logger.info("=== Progressive Training Demo ===")
    
    from training.progressive_training import (
        ProgressiveResolutionScheduler, ResolutionTransform,
        create_default_progressive_schedule
    )
    
    # Create progressive schedule
    base_resolution = (32, 32)
    resolution_schedule, epochs_per_stage = create_default_progressive_schedule(
        base_resolution, n_stages=3
    )
    
    logger.info(f"Progressive schedule: {resolution_schedule}")
    logger.info(f"Epochs per stage: {epochs_per_stage}")
    
    # Test resolution scheduler
    scheduler = ProgressiveResolutionScheduler(resolution_schedule, epochs_per_stage)
    
    logger.info("Simulating progressive training...")
    for epoch in range(sum(epochs_per_stage)):
        current_res = scheduler.get_current_resolution()
        current_stage = scheduler.get_current_stage()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Stage {current_stage}, Resolution {current_res}")
        
        stage_changed = scheduler.step()
        if stage_changed:
            logger.info(f"  -> Advanced to stage {scheduler.get_current_stage()}")
    
    # Test resolution transform
    logger.info("Testing resolution transforms...")
    transform_down = ResolutionTransform((8, 8), method='nearest')
    transform_up = ResolutionTransform((32, 32), method='bilinear')
    
    test_tensor = torch.randn(1, 1, 16, 16)
    downsampled = transform_down(test_tensor)
    upsampled = transform_up(downsampled)
    
    logger.info(f"Original: {test_tensor.shape} -> Downsampled: {downsampled.shape} -> Upsampled: {upsampled.shape}")
    
    logger.info("Progressive training demo completed!\n")


def demo_ensemble_concepts():
    """Demonstrate ensemble training concepts."""
    logger.info("=== Ensemble Training Demo ===")
    
    # Since ensemble training requires full models, we'll just demonstrate the concepts
    logger.info("Ensemble training provides:")
    logger.info("  - Multiple models with different random initializations")
    logger.info("  - Uncertainty estimation through prediction variance")
    logger.info("  - Improved robustness and physics consistency")
    logger.info("  - Model diversity metrics")
    
    # Simulate ensemble predictions
    n_members = 5
    n_samples = 100
    latent_dim = 2
    
    # Simulate different model predictions
    predictions = []
    for member in range(n_members):
        torch.manual_seed(42 + member)  # Different seeds
        pred = torch.randn(n_samples, latent_dim)
        predictions.append(pred)
    
    predictions = torch.stack(predictions, dim=0)  # [n_members, n_samples, latent_dim]
    
    # Calculate ensemble statistics
    ensemble_mean = torch.mean(predictions, dim=0)
    ensemble_std = torch.std(predictions, dim=0)
    epistemic_uncertainty = torch.var(predictions, dim=0)
    
    logger.info(f"Ensemble predictions shape: {predictions.shape}")
    logger.info(f"Mean prediction std: {torch.mean(ensemble_std).item():.4f}")
    logger.info(f"Mean epistemic uncertainty: {torch.mean(epistemic_uncertainty).item():.4f}")
    
    logger.info("Ensemble training demo completed!\n")


def main():
    """Run all demonstrations."""
    logger.info("Enhanced Training Features Demonstration")
    logger.info("=" * 50)
    
    try:
        demo_advanced_schedulers()
        demo_data_augmentation()
        demo_progressive_training()
        demo_ensemble_concepts()
        
        logger.info("All demonstrations completed successfully!")
        logger.info("\nThese enhanced training features provide:")
        logger.info("  ✓ Advanced learning rate scheduling for better convergence")
        logger.info("  ✓ Physics-aware data augmentation for improved generalization")
        logger.info("  ✓ Progressive training for hierarchical learning")
        logger.info("  ✓ Ensemble methods for uncertainty quantification")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()