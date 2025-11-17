#!/usr/bin/env python3
"""
Train final enhanced VAE with stronger critical enhancement for Task 5.3.

Task 5.2 (mag correlation ≥96%) is passing with 99.86%.
Task 5.3 (variance ratio >2.0) needs improvement - currently 1.06.

Solution: Increase critical_enhancement weight from 1.0 to 3.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.train_enhanced_vae import EnhancedVAETrainer
from src.training.enhanced_physics_vae import EnhancedPhysicsLossWeights
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


def main():
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    logger.info("Training final enhanced VAE with stronger critical enhancement")
    
    # Create trainer with custom loss weights
    trainer = EnhancedVAETrainer(model_type='3d')
    
    # Increase critical enhancement weight to improve Task 5.3
    trainer.loss_weights = EnhancedPhysicsLossWeights(
        reconstruction=1.0,
        kl_divergence=1.0,
        magnetization_correlation=2.0,  # Keep at 2.0 (Task 5.2 passing)
        energy_consistency=1.0,
        temperature_ordering=1.5,        # Keep at 1.5
        critical_enhancement=3.0          # Increase from 1.0 to 3.0 for Task 5.3
    )
    
    logger.info(f"Loss weights: mag_corr={trainer.loss_weights.magnetization_correlation}, "
                f"temp_order={trainer.loss_weights.temperature_ordering}, "
                f"critical_enh={trainer.loss_weights.critical_enhancement}")
    
    # Train model
    model, history, quality = trainer.train_enhanced_vae(
        data_path='data/vae_training_3d_ising.h5',
        output_dir='models/enhanced_3d_vae_final_v2',
        latent_dim=2
    )
    
    logger.info("Training completed")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL ENHANCED VAE TRAINING SUMMARY")
    print("="*60)
    print(f"Model quality: {quality['quality_level']} ({quality['quality_score']:.1f}/100)")
    print(f"Magnetization correlation: {quality['max_mag_correlation']:.4f} (target: ≥0.96)")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print("="*60)
    print("\nNext step: Run validate_enhanced_vae.py to check Task 5.3")


if __name__ == "__main__":
    main()
