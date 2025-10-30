"""
Enhanced Validation Configuration System

This module provides configuration management for enhanced physics validation features,
including validation levels, feature toggles, and performance optimization settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation comprehensiveness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class PerformanceMode(Enum):
    """Performance optimization modes."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass
class CriticalExponentConfig:
    """Configuration for critical exponent validation."""
    enable: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    deviation_threshold: float = 0.1
    universality_classes: List[str] = field(default_factory=lambda: ["ising_2d", "ising_3d", "xy_2d", "heisenberg_3d"])


@dataclass
class SymmetryValidationConfig:
    """Configuration for symmetry validation."""
    enable: bool = True
    hamiltonian_symmetries: List[str] = field(default_factory=lambda: ["Z2"])
    consistency_threshold: float = 0.8
    check_broken_symmetries: bool = True


@dataclass
class FiniteSizeScalingConfig:
    """Configuration for finite-size scaling validation."""
    enable: bool = False  # Requires multi-size data
    min_system_sizes: int = 3
    scaling_collapse_threshold: float = 0.8
    correlation_length_method: str = "exponential_fit"


@dataclass
class TheoreticalModelConfig:
    """Configuration for theoretical model validation."""
    enable: bool = True
    models_to_validate: List[str] = field(default_factory=lambda: ["ising"])
    dimensionality: int = 2
    system_size: int = 32
    coupling_strength: float = 1.0
    include_finite_size_corrections: bool = True


@dataclass
class StatisticalAnalysisConfig:
    """Configuration for statistical physics analysis."""
    enable: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    ensemble_analysis: bool = False  # Requires ensemble data
    hypothesis_testing: bool = True
    phase_boundary_uncertainty: bool = True
    random_seed: Optional[int] = None


@dataclass
class ExperimentalComparisonConfig:
    """Configuration for experimental benchmark comparison."""
    enable: bool = False  # Optional feature
    benchmark_datasets: List[str] = field(default_factory=lambda: ["ising_2d_onsager"])
    agreement_threshold: float = 0.8
    meta_analysis: bool = True
    custom_benchmark_path: Optional[str] = None


@dataclass
class ReportGenerationConfig:
    """Configuration for physics review report generation."""
    enable: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    include_educational_content: bool = True
    include_visualizations: bool = True
    violation_severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical_exponent_deviation": 0.1,
        "symmetry_consistency": 0.8,
        "scaling_collapse_quality": 0.8
    })
    output_format: str = "dict"  # "dict", "json", "yaml"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    mode: PerformanceMode = PerformanceMode.BALANCED
    parallel_processing: bool = True
    n_jobs: int = -1  # -1 for all available cores
    memory_limit_gb: Optional[float] = None
    cache_results: bool = True
    early_stopping: bool = False


@dataclass
class EnhancedValidationConfig:
    """
    Comprehensive configuration for enhanced physics validation.
    
    This class provides a centralized configuration system for all enhanced
    validation features with sensible defaults and validation level presets.
    """
    
    # Main configuration
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_enhanced_features: bool = True
    
    # Component configurations
    critical_exponents: CriticalExponentConfig = field(default_factory=CriticalExponentConfig)
    symmetry_validation: SymmetryValidationConfig = field(default_factory=SymmetryValidationConfig)
    finite_size_scaling: FiniteSizeScalingConfig = field(default_factory=FiniteSizeScalingConfig)
    theoretical_models: TheoreticalModelConfig = field(default_factory=TheoreticalModelConfig)
    statistical_analysis: StatisticalAnalysisConfig = field(default_factory=StatisticalAnalysisConfig)
    experimental_comparison: ExperimentalComparisonConfig = field(default_factory=ExperimentalComparisonConfig)
    report_generation: ReportGenerationConfig = field(default_factory=ReportGenerationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Legacy compatibility
    legacy_compatibility: bool = True
    
    def __post_init__(self):
        """Apply validation level presets after initialization."""
        self.apply_validation_level_preset()
    
    def apply_validation_level_preset(self):
        """Apply preset configurations based on validation level."""
        if self.validation_level == ValidationLevel.BASIC:
            self._apply_basic_preset()
        elif self.validation_level == ValidationLevel.STANDARD:
            self._apply_standard_preset()
        elif self.validation_level == ValidationLevel.COMPREHENSIVE:
            self._apply_comprehensive_preset()
    
    def _apply_basic_preset(self):
        """Apply basic validation preset - minimal features for speed."""
        self.enable_enhanced_features = True
        
        # Enable only essential features
        self.critical_exponents.enable = True
        self.critical_exponents.bootstrap_samples = 500
        
        self.symmetry_validation.enable = True
        
        self.finite_size_scaling.enable = False
        
        self.theoretical_models.enable = True
        self.theoretical_models.models_to_validate = ["ising"]
        
        self.statistical_analysis.enable = True
        self.statistical_analysis.bootstrap_samples = 500
        self.statistical_analysis.ensemble_analysis = False
        
        self.experimental_comparison.enable = False
        
        self.report_generation.enable = True
        self.report_generation.include_educational_content = False
        self.report_generation.include_visualizations = False
        
        self.performance.mode = PerformanceMode.FAST
    
    def _apply_standard_preset(self):
        """Apply standard validation preset - balanced features and performance."""
        self.enable_enhanced_features = True
        
        # Standard feature set
        self.critical_exponents.enable = True
        self.critical_exponents.bootstrap_samples = 1000
        
        self.symmetry_validation.enable = True
        
        self.finite_size_scaling.enable = False  # Requires special data
        
        self.theoretical_models.enable = True
        self.theoretical_models.models_to_validate = ["ising"]
        
        self.statistical_analysis.enable = True
        self.statistical_analysis.bootstrap_samples = 1000
        self.statistical_analysis.ensemble_analysis = False  # Requires special data
        
        self.experimental_comparison.enable = False  # Optional
        
        self.report_generation.enable = True
        self.report_generation.include_educational_content = True
        self.report_generation.include_visualizations = True
        
        self.performance.mode = PerformanceMode.BALANCED
    
    def _apply_comprehensive_preset(self):
        """Apply comprehensive validation preset - all features enabled."""
        self.enable_enhanced_features = True
        
        # Enable all features
        self.critical_exponents.enable = True
        self.critical_exponents.bootstrap_samples = 2000
        
        self.symmetry_validation.enable = True
        
        self.finite_size_scaling.enable = True
        
        self.theoretical_models.enable = True
        self.theoretical_models.models_to_validate = ["ising", "xy", "heisenberg"]
        
        self.statistical_analysis.enable = True
        self.statistical_analysis.bootstrap_samples = 2000
        self.statistical_analysis.ensemble_analysis = True
        
        self.experimental_comparison.enable = True
        self.experimental_comparison.meta_analysis = True
        
        self.report_generation.enable = True
        self.report_generation.include_educational_content = True
        self.report_generation.include_visualizations = True
        
        self.performance.mode = PerformanceMode.THOROUGH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name, field_def in obj.__dataclass_fields__.items():
                    value = getattr(obj, field_name)
                    if hasattr(value, '__dataclass_fields__'):
                        result[field_name] = _dataclass_to_dict(value)
                    elif isinstance(value, Enum):
                        result[field_name] = value.value
                    elif isinstance(value, (list, dict)):
                        result[field_name] = value
                    else:
                        result[field_name] = value
                return result
            else:
                return obj
        
        return _dataclass_to_dict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedValidationConfig':
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'validation_level' in config_dict:
            if isinstance(config_dict['validation_level'], str):
                config_dict['validation_level'] = ValidationLevel(config_dict['validation_level'])
        
        if 'performance' in config_dict and 'mode' in config_dict['performance']:
            if isinstance(config_dict['performance']['mode'], str):
                config_dict['performance']['mode'] = PerformanceMode(config_dict['performance']['mode'])
        
        # Create nested dataclass instances
        nested_configs = {}
        
        if 'critical_exponents' in config_dict:
            nested_configs['critical_exponents'] = CriticalExponentConfig(**config_dict['critical_exponents'])
        
        if 'symmetry_validation' in config_dict:
            nested_configs['symmetry_validation'] = SymmetryValidationConfig(**config_dict['symmetry_validation'])
        
        if 'finite_size_scaling' in config_dict:
            nested_configs['finite_size_scaling'] = FiniteSizeScalingConfig(**config_dict['finite_size_scaling'])
        
        if 'theoretical_models' in config_dict:
            nested_configs['theoretical_models'] = TheoreticalModelConfig(**config_dict['theoretical_models'])
        
        if 'statistical_analysis' in config_dict:
            nested_configs['statistical_analysis'] = StatisticalAnalysisConfig(**config_dict['statistical_analysis'])
        
        if 'experimental_comparison' in config_dict:
            nested_configs['experimental_comparison'] = ExperimentalComparisonConfig(**config_dict['experimental_comparison'])
        
        if 'report_generation' in config_dict:
            nested_configs['report_generation'] = ReportGenerationConfig(**config_dict['report_generation'])
        
        if 'performance' in config_dict:
            nested_configs['performance'] = PerformanceConfig(**config_dict['performance'])
        
        # Create main config with nested configs
        main_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in nested_configs}
        main_config_dict.update(nested_configs)
        
        return cls(**main_config_dict)
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to file (YAML or JSON based on extension)."""
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'EnhancedValidationConfig':
        """Load configuration from file (YAML or JSON)."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Check for conflicting settings
        if not self.enable_enhanced_features:
            if (self.critical_exponents.enable or self.symmetry_validation.enable or 
                self.theoretical_models.enable or self.statistical_analysis.enable):
                warnings.append("Enhanced features disabled but individual components enabled")
        
        # Check for data requirements
        if self.finite_size_scaling.enable:
            warnings.append("Finite-size scaling requires multi-size data to be provided")
        
        if self.statistical_analysis.ensemble_analysis:
            warnings.append("Ensemble analysis requires ensemble data to be provided")
        
        if self.experimental_comparison.enable and not self.experimental_comparison.benchmark_datasets:
            warnings.append("Experimental comparison enabled but no benchmark datasets specified")
        
        # Check performance settings
        if self.performance.n_jobs == -1 and self.performance.mode == PerformanceMode.FAST:
            warnings.append("Using all CPU cores in fast mode may not be optimal")
        
        # Check bootstrap sample counts
        if self.critical_exponents.bootstrap_samples < 100:
            warnings.append("Very low bootstrap sample count may give unreliable confidence intervals")
        
        return warnings
    
    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration dictionary compatible with legacy validation methods.
        
        Returns:
            Dictionary with keys expected by the enhanced comprehensive_physics_validation method
        """
        return {
            'validation_level': self.validation_level.value,
            'enable_enhanced_features': self.enable_enhanced_features,
            'enable_theoretical_validation': self.theoretical_models.enable,
            'enable_statistical_analysis': self.statistical_analysis.enable,
            'enable_experimental_comparison': self.experimental_comparison.enable,
            'enable_report_generation': self.report_generation.enable,
            
            # Specific parameters
            'dimensionality': self.theoretical_models.dimensionality,
            'system_size': self.theoretical_models.system_size,
            'hamiltonian_symmetries': self.symmetry_validation.hamiltonian_symmetries,
            'confidence_level': self.statistical_analysis.confidence_level,
            'n_bootstrap': self.statistical_analysis.bootstrap_samples,
            'benchmark_datasets': self.experimental_comparison.benchmark_datasets,
            'model_type': self.theoretical_models.models_to_validate[0] if self.theoretical_models.models_to_validate else 'ising',
            'include_educational_content': self.report_generation.include_educational_content,
            'report_validation_level': self.report_generation.validation_level.value
        }


class EnhancedValidationConfigManager:
    """
    Manager class for enhanced validation configurations.
    
    Provides utilities for creating, loading, and managing validation configurations
    with preset templates and validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._preset_configs = {}
        self._initialize_presets()
    
    def _initialize_presets(self):
        """Initialize preset configurations."""
        # Basic preset
        basic_config = EnhancedValidationConfig(validation_level=ValidationLevel.BASIC)
        self._preset_configs['basic'] = basic_config
        
        # Standard preset
        standard_config = EnhancedValidationConfig(validation_level=ValidationLevel.STANDARD)
        self._preset_configs['standard'] = standard_config
        
        # Comprehensive preset
        comprehensive_config = EnhancedValidationConfig(validation_level=ValidationLevel.COMPREHENSIVE)
        self._preset_configs['comprehensive'] = comprehensive_config
        
        # Publication-ready preset
        publication_config = EnhancedValidationConfig(validation_level=ValidationLevel.COMPREHENSIVE)
        publication_config.experimental_comparison.enable = True
        publication_config.report_generation.include_educational_content = True
        publication_config.report_generation.include_visualizations = True
        publication_config.statistical_analysis.bootstrap_samples = 5000
        self._preset_configs['publication'] = publication_config
        
        # Fast debugging preset
        debug_config = EnhancedValidationConfig(validation_level=ValidationLevel.BASIC)
        debug_config.critical_exponents.bootstrap_samples = 100
        debug_config.statistical_analysis.bootstrap_samples = 100
        debug_config.performance.mode = PerformanceMode.FAST
        debug_config.report_generation.include_visualizations = False
        self._preset_configs['debug'] = debug_config
    
    def get_preset_config(self, preset_name: str) -> EnhancedValidationConfig:
        """Get a preset configuration by name."""
        if preset_name not in self._preset_configs:
            available_presets = list(self._preset_configs.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
        
        # Return a copy to avoid modifying the original
        import copy
        return copy.deepcopy(self._preset_configs[preset_name])
    
    def list_presets(self) -> List[str]:
        """List available preset configuration names."""
        return list(self._preset_configs.keys())
    
    def create_custom_config(self, 
                           base_preset: str = 'standard',
                           **overrides) -> EnhancedValidationConfig:
        """
        Create a custom configuration based on a preset with overrides.
        
        Args:
            base_preset: Name of base preset to start from
            **overrides: Configuration overrides
            
        Returns:
            Custom configuration
        """
        config = self.get_preset_config(base_preset)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
        
        return config
    
    def validate_and_warn(self, config: EnhancedValidationConfig) -> EnhancedValidationConfig:
        """Validate configuration and log warnings."""
        warnings = config.validate_config()
        
        for warning in warnings:
            self.logger.warning(f"Configuration warning: {warning}")
        
        return config


# Global configuration manager instance
config_manager = EnhancedValidationConfigManager()


def get_default_config(validation_level: str = 'standard') -> EnhancedValidationConfig:
    """
    Get default configuration for specified validation level.
    
    Args:
        validation_level: Validation level ('basic', 'standard', 'comprehensive')
        
    Returns:
        Default configuration for the specified level
    """
    return config_manager.get_preset_config(validation_level)


def create_config_from_dict(config_dict: Dict[str, Any]) -> EnhancedValidationConfig:
    """Create configuration from dictionary with validation."""
    config = EnhancedValidationConfig.from_dict(config_dict)
    return config_manager.validate_and_warn(config)