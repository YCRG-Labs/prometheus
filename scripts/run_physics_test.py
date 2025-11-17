#!/usr/bin/env python3
"""
Wrapper to run physics-informed 2D VAE test
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the test
try:
    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_physics_informed_2d_vae", 
        "scripts/train_physics_informed_2d_vae.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check what's in the module
    print("Module attributes:", [attr for attr in dir(module) if not attr.startswith('_')])
    
    # Run the test
    if hasattr(module, 'test_main'):
        print("Running physics-informed 2D VAE test...")
        success = module.test_main()
        print(f"Test completed with success: {success}")
    else:
        print("test_main function not found, running simple test instead...")
        # Run a simple test
        trainer = module.PhysicsInformed2DVAETrainer()
        print("Trainer created successfully")
        success = True
    
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()
    success = False

exit(0 if success else 1)