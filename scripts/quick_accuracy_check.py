#!/usr/bin/env python3
"""
Quick accuracy check to see current performance.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.comprehensive_accuracy_assessment import (
        create_synthetic_high_quality_data,
        create_realistic_vae_representation,
        test_vae_accuracy,
        test_raw_magnetization_accuracy
    )
    
    print("=" * 60)
    print("QUICK ACCURACY CHECK")
    print("=" * 60)
    
    # Create synthetic data
    print("Creating synthetic high-quality data...")
    temperatures, magnetizations, energies, tc, beta, nu = create_synthetic_high_quality_data(500)
    
    theoretical_values = {
        'tc': tc,
        'beta': beta,
        'nu': nu
    }
    
    print(f"Generated {len(temperatures)} samples")
    print(f"Theoretical Tc: {tc:.3f}, β: {beta:.3f}, ν: {nu:.3f}")
    
    # Create VAE representation
    print("Creating realistic VAE representation...")
    latent_repr = create_realistic_vae_representation(temperatures, magnetizations, energies, tc)
    
    # Test VAE accuracy
    print("Testing VAE-based accuracy...")
    try:
        vae_results = test_vae_accuracy(latent_repr, theoretical_values)
        vae_accuracy = vae_results.get('overall_accuracy_percent', 0)
        print(f"✓ VAE Overall Accuracy: {vae_accuracy:.1f}%")
    except Exception as e:
        print(f"❌ VAE test failed: {e}")
        vae_accuracy = 0
    
    # Test raw magnetization accuracy
    print("Testing raw magnetization accuracy...")
    try:
        raw_results = test_raw_magnetization_accuracy(latent_repr, theoretical_values)
        raw_accuracy = raw_results.get('overall_accuracy_percent', 0)
        print(f"✓ Raw Magnetization Accuracy: {raw_accuracy:.1f}%")
    except Exception as e:
        print(f"❌ Raw magnetization test failed: {e}")
        raw_accuracy = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    
    best_accuracy = max(vae_accuracy, raw_accuracy)
    print(f"Best Overall Accuracy: {best_accuracy:.1f}%")
    
    if best_accuracy >= 70:
        print("✅ MEETS PUBLICATION STANDARDS (≥70%)")
    elif best_accuracy >= 50:
        print("⚠️  ACCEPTABLE PERFORMANCE (50-70%)")
    elif best_accuracy >= 30:
        print("❌ POOR PERFORMANCE (30-50%)")
    else:
        print("❌ VERY POOR PERFORMANCE (<30%)")
    
    # Improvement needed
    target = 75
    if best_accuracy < target:
        gap = target - best_accuracy
        print(f"Gap to target ({target}%): {gap:.1f}%")
        months_needed = max(1, int(gap / 10))
        print(f"Estimated improvement time: {months_needed} months")
    
    print(f"\nBetter method: {'VAE' if vae_accuracy > raw_accuracy else 'Raw Magnetization'}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback accuracy assessment...")
    
    # Fallback: Read from existing assessment
    try:
        assessment_file = Path("CURRENT_MODEL_ACCURACY_ASSESSMENT.md")
        if assessment_file.exists():
            content = assessment_file.read_text()
            if "8.8% overall" in content:
                print("Current documented accuracy: 8.8% overall")
                print("❌ VERY POOR PERFORMANCE - Major improvements needed")
            else:
                print("Could not parse current accuracy from assessment file")
        else:
            print("No current assessment file found")
    except Exception as e:
        print(f"Error reading assessment: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
    print("Current system needs significant accuracy improvements")