"""
Data Augmentation Techniques for Ising Model Configurations

This module implements physics-aware data augmentation techniques including
rotations, reflections, and spin transformations that preserve the physical
properties of Ising model configurations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable, Union, Dict
import random
import logging


class IsingAugmentation:
    """
    Base class for Ising model data augmentation.
    
    Provides physics-aware transformations that preserve the statistical
    properties and symmetries of the 2D Ising model.
    """
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentation.
        
        Args:
            probability: Probability of applying the augmentation
        """
        self.probability = probability
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation with given probability."""
        if random.random() < self.probability:
            return self.transform(x)
        return x
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation. To be implemented by subclasses."""
        raise NotImplementedError


class RotationAugmentation(IsingAugmentation):
    """
    Rotation augmentation for 2D Ising configurations.
    
    Applies 90°, 180°, or 270° rotations which preserve the Ising model
    symmetries and physical properties.
    """
    
    def __init__(self, 
                 angles: List[int] = [90, 180, 270],
                 probability: float = 0.5):
        """
        Initialize rotation augmentation.
        
        Args:
            angles: List of rotation angles (in degrees)
            probability: Probability of applying rotation
        """
        super().__init__(probability)
        self.angles = angles
        
        # Validate angles
        valid_angles = [90, 180, 270]
        for angle in angles:
            if angle not in valid_angles:
                raise ValueError(f"Invalid rotation angle: {angle}. Must be one of {valid_angles}")
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        angle = random.choice(self.angles)
        
        if angle == 90:
            # 90° clockwise rotation
            return torch.rot90(x, k=-1, dims=(-2, -1))
        elif angle == 180:
            # 180° rotation
            return torch.rot90(x, k=2, dims=(-2, -1))
        elif angle == 270:
            # 270° clockwise rotation (90° counter-clockwise)
            return torch.rot90(x, k=1, dims=(-2, -1))
        else:
            return x


class ReflectionAugmentation(IsingAugmentation):
    """
    Reflection augmentation for 2D Ising configurations.
    
    Applies horizontal and/or vertical reflections which preserve
    the Ising model symmetries.
    """
    
    def __init__(self, 
                 horizontal: bool = True,
                 vertical: bool = True,
                 probability: float = 0.5):
        """
        Initialize reflection augmentation.
        
        Args:
            horizontal: Whether to allow horizontal reflections
            vertical: Whether to allow vertical reflections
            probability: Probability of applying reflection
        """
        super().__init__(probability)
        self.horizontal = horizontal
        self.vertical = vertical
        
        if not (horizontal or vertical):
            raise ValueError("At least one of horizontal or vertical must be True")
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random reflection."""
        transformations = []
        
        if self.horizontal:
            transformations.append('horizontal')
        if self.vertical:
            transformations.append('vertical')
        
        # Randomly choose transformation
        transform_type = random.choice(transformations)
        
        if transform_type == 'horizontal':
            # Horizontal flip (left-right)
            return torch.flip(x, dims=[-1])
        elif transform_type == 'vertical':
            # Vertical flip (up-down)
            return torch.flip(x, dims=[-2])
        else:
            return x


class SpinFlipAugmentation(IsingAugmentation):
    """
    Global spin flip augmentation.
    
    Applies global spin inversion (all +1 → -1, all -1 → +1) which
    preserves the Ising model physics due to Z2 symmetry.
    """
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize spin flip augmentation.
        
        Args:
            probability: Probability of applying global spin flip
        """
        super().__init__(probability)
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global spin flip."""
        return -x


class CompositeAugmentation(IsingAugmentation):
    """
    Composite augmentation that combines multiple transformations.
    
    Applies a sequence of augmentations with individual probabilities.
    """
    
    def __init__(self, 
                 augmentations: List[IsingAugmentation],
                 probability: float = 1.0):
        """
        Initialize composite augmentation.
        
        Args:
            augmentations: List of augmentation objects to apply
            probability: Overall probability of applying any augmentation
        """
        super().__init__(probability)
        self.augmentations = augmentations
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequence of augmentations."""
        result = x
        for aug in self.augmentations:
            result = aug(result)
        return result


class RandomAugmentation(IsingAugmentation):
    """
    Random augmentation that selects one transformation from a list.
    
    Randomly chooses and applies one augmentation from the provided list.
    """
    
    def __init__(self, 
                 augmentations: List[IsingAugmentation],
                 probability: float = 0.5):
        """
        Initialize random augmentation.
        
        Args:
            augmentations: List of augmentation objects to choose from
            probability: Probability of applying an augmentation
        """
        super().__init__(probability)
        self.augmentations = augmentations
        
        if not augmentations:
            raise ValueError("At least one augmentation must be provided")
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomly selected augmentation."""
        aug = random.choice(self.augmentations)
        return aug.transform(x)  # Call transform directly to avoid double probability check


class NoiseAugmentation(IsingAugmentation):
    """
    Noise augmentation for Ising configurations.
    
    Adds small amounts of Gaussian noise to simulate thermal fluctuations
    while preserving the overall spin structure.
    """
    
    def __init__(self, 
                 noise_std: float = 0.1,
                 probability: float = 0.5):
        """
        Initialize noise augmentation.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            probability: Probability of applying noise
        """
        super().__init__(probability)
        self.noise_std = noise_std
        
        if noise_std <= 0:
            raise ValueError("Noise standard deviation must be positive")
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise


class TemperatureScalingAugmentation(IsingAugmentation):
    """
    Temperature-aware scaling augmentation.
    
    Applies slight scaling to simulate different effective temperatures
    while maintaining the spin configuration structure.
    """
    
    def __init__(self, 
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 probability: float = 0.3):
        """
        Initialize temperature scaling augmentation.
        
        Args:
            scale_range: Range of scaling factors (min, max)
            probability: Probability of applying scaling
        """
        super().__init__(probability)
        self.scale_range = scale_range
        
        if scale_range[0] >= scale_range[1]:
            raise ValueError("Scale range must have min < max")
        if scale_range[0] <= 0:
            raise ValueError("Scale factors must be positive")
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale_factor


def create_standard_augmentation(rotation_prob: float = 0.7,
                               reflection_prob: float = 0.5,
                               spin_flip_prob: float = 0.3,
                               noise_prob: float = 0.2,
                               noise_std: float = 0.05) -> CompositeAugmentation:
    """
    Create a standard augmentation pipeline for Ising configurations.
    
    Args:
        rotation_prob: Probability of rotation augmentation
        reflection_prob: Probability of reflection augmentation
        spin_flip_prob: Probability of spin flip augmentation
        noise_prob: Probability of noise augmentation
        noise_std: Standard deviation for noise augmentation
        
    Returns:
        Composite augmentation with standard transformations
    """
    augmentations = [
        RotationAugmentation(probability=rotation_prob),
        ReflectionAugmentation(probability=reflection_prob),
        SpinFlipAugmentation(probability=spin_flip_prob),
        NoiseAugmentation(noise_std=noise_std, probability=noise_prob)
    ]
    
    return CompositeAugmentation(augmentations, probability=1.0)


def create_conservative_augmentation(rotation_prob: float = 0.5,
                                   reflection_prob: float = 0.3,
                                   spin_flip_prob: float = 0.2) -> CompositeAugmentation:
    """
    Create a conservative augmentation pipeline with only physics-preserving transformations.
    
    Args:
        rotation_prob: Probability of rotation augmentation
        reflection_prob: Probability of reflection augmentation
        spin_flip_prob: Probability of spin flip augmentation
        
    Returns:
        Conservative composite augmentation
    """
    augmentations = [
        RotationAugmentation(probability=rotation_prob),
        ReflectionAugmentation(probability=reflection_prob),
        SpinFlipAugmentation(probability=spin_flip_prob)
    ]
    
    return CompositeAugmentation(augmentations, probability=1.0)


def create_aggressive_augmentation(rotation_prob: float = 0.8,
                                 reflection_prob: float = 0.7,
                                 spin_flip_prob: float = 0.5,
                                 noise_prob: float = 0.4,
                                 noise_std: float = 0.1,
                                 scale_prob: float = 0.3) -> CompositeAugmentation:
    """
    Create an aggressive augmentation pipeline for improved generalization.
    
    Args:
        rotation_prob: Probability of rotation augmentation
        reflection_prob: Probability of reflection augmentation
        spin_flip_prob: Probability of spin flip augmentation
        noise_prob: Probability of noise augmentation
        noise_std: Standard deviation for noise augmentation
        scale_prob: Probability of scaling augmentation
        
    Returns:
        Aggressive composite augmentation
    """
    augmentations = [
        RotationAugmentation(probability=rotation_prob),
        ReflectionAugmentation(probability=reflection_prob),
        SpinFlipAugmentation(probability=spin_flip_prob),
        NoiseAugmentation(noise_std=noise_std, probability=noise_prob),
        TemperatureScalingAugmentation(probability=scale_prob)
    ]
    
    return CompositeAugmentation(augmentations, probability=1.0)


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies augmentation on-the-fly.
    
    Wraps an existing dataset and applies augmentation transformations
    during data loading for improved training efficiency.
    """
    
    def __init__(self, 
                 base_dataset: torch.utils.data.Dataset,
                 augmentation: Optional[IsingAugmentation] = None,
                 augment_probability: float = 0.5):
        """
        Initialize augmented dataset.
        
        Args:
            base_dataset: Base dataset to wrap
            augmentation: Augmentation to apply
            augment_probability: Probability of applying augmentation to each sample
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.augment_probability = augment_probability
        self.logger = logging.getLogger(__name__)
        
        if augmentation is None:
            self.augmentation = create_standard_augmentation()
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple]:
        """Get augmented dataset item."""
        item = self.base_dataset[idx]
        
        # Handle both single tensor and tuple returns
        if isinstance(item, tuple):
            data, *rest = item
            
            # Apply augmentation with probability
            if random.random() < self.augment_probability:
                data = self.augmentation(data)
            
            return (data, *rest)
        else:
            # Single tensor case
            if random.random() < self.augment_probability:
                item = self.augmentation(item)
            
            return item
    
    def set_augmentation_probability(self, probability: float) -> None:
        """Set augmentation probability."""
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self.augment_probability = probability
        self.logger.info(f"Augmentation probability set to {probability}")
    
    def disable_augmentation(self) -> None:
        """Disable augmentation."""
        self.set_augmentation_probability(0.0)
    
    def enable_augmentation(self) -> None:
        """Enable augmentation with default probability."""
        self.set_augmentation_probability(0.5)


def test_augmentation_physics_preservation(augmentation: IsingAugmentation,
                                         test_config: torch.Tensor,
                                         n_tests: int = 100) -> Dict[str, float]:
    """
    Test that augmentation preserves physical properties.
    
    Args:
        augmentation: Augmentation to test
        test_config: Test configuration tensor
        n_tests: Number of tests to run
        
    Returns:
        Dictionary with preservation statistics
    """
    original_energy = torch.sum(test_config).item()  # Simple energy proxy
    original_magnetization = torch.mean(test_config).item()
    
    energy_diffs = []
    mag_diffs = []
    
    for _ in range(n_tests):
        augmented = augmentation.transform(test_config.clone())
        
        aug_energy = torch.sum(augmented).item()
        aug_magnetization = torch.mean(augmented).item()
        
        energy_diffs.append(abs(aug_energy - original_energy))
        mag_diffs.append(abs(aug_magnetization - abs(original_magnetization)))  # Absolute for spin flip
    
    return {
        'mean_energy_diff': np.mean(energy_diffs),
        'std_energy_diff': np.std(energy_diffs),
        'mean_mag_diff': np.mean(mag_diffs),
        'std_mag_diff': np.std(mag_diffs),
        'max_energy_diff': np.max(energy_diffs),
        'max_mag_diff': np.max(mag_diffs)
    }