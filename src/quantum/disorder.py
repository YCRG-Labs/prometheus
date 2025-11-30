"""
Disorder infrastructure for quantum spin systems.

Implements:
- Random coupling generators (box, Gaussian distributions)
- Disorder averaging framework
- Parallel disorder realization computation
- Statistical analysis (mean, std, distributions)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing


class DisorderType(Enum):
    """Types of disorder distributions."""
    BOX = "box"  # Uniform distribution [center - width/2, center + width/2]
    GAUSSIAN = "gaussian"  # Normal distribution N(mean, std)
    BINARY = "binary"  # Binary disorder {-1, +1} or {0, 1}
    LOGUNIFORM = "loguniform"  # Log-uniform distribution


@dataclass
class DisorderConfig:
    """Configuration for disorder in a quantum system."""
    disorder_type: DisorderType
    center: float = 1.0  # Mean/center value
    width: float = 0.0  # Width for box, std for Gaussian
    seed: Optional[int] = None  # Random seed for reproducibility
    
    def __post_init__(self):
        if self.width < 0:
            raise ValueError("Disorder width must be non-negative")


@dataclass
class DisorderRealization:
    """A single realization of disorder."""
    couplings: np.ndarray  # J_i values
    transverse_fields: np.ndarray  # h_i values
    longitudinal_fields: np.ndarray  # hz_i values
    seed: int  # Seed used for this realization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'couplings': self.couplings.tolist(),
            'transverse_fields': self.transverse_fields.tolist(),
            'longitudinal_fields': self.longitudinal_fields.tolist(),
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisorderRealization':
        """Create from dictionary."""
        return cls(
            couplings=np.array(data['couplings']),
            transverse_fields=np.array(data['transverse_fields']),
            longitudinal_fields=np.array(data['longitudinal_fields']),
            seed=data['seed']
        )


class RandomCouplingGenerator:
    """
    Generate random couplings for disordered quantum systems.
    
    Supports box (uniform) and Gaussian distributions for:
    - Exchange couplings J_i
    - Transverse fields h_i
    - Longitudinal fields hz_i
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    def set_seed(self, seed: int) -> None:
        """Set new random seed."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    def generate_box(
        self,
        size: int,
        center: float = 1.0,
        width: float = 0.5
    ) -> np.ndarray:
        """
        Generate random values from box (uniform) distribution.
        
        Values are drawn from [center - width/2, center + width/2].
        
        Args:
            size: Number of values to generate
            center: Center of the distribution
            width: Width of the distribution
            
        Returns:
            Array of random values
        """
        if width == 0:
            return np.full(size, center)
        
        low = center - width / 2
        high = center + width / 2
        return self._rng.uniform(low, high, size)
    
    def generate_gaussian(
        self,
        size: int,
        mean: float = 1.0,
        std: float = 0.5
    ) -> np.ndarray:
        """
        Generate random values from Gaussian distribution.
        
        Args:
            size: Number of values to generate
            mean: Mean of the distribution
            std: Standard deviation
            
        Returns:
            Array of random values
        """
        if std == 0:
            return np.full(size, mean)
        
        return self._rng.normal(mean, std, size)
    
    def generate_binary(
        self,
        size: int,
        values: Tuple[float, float] = (-1.0, 1.0),
        p: float = 0.5
    ) -> np.ndarray:
        """
        Generate binary random values.
        
        Args:
            size: Number of values to generate
            values: Tuple of (value_0, value_1)
            p: Probability of value_1
            
        Returns:
            Array of random values
        """
        choices = self._rng.random(size) < p
        return np.where(choices, values[1], values[0])
    
    def generate_loguniform(
        self,
        size: int,
        low: float = 0.1,
        high: float = 10.0
    ) -> np.ndarray:
        """
        Generate random values from log-uniform distribution.
        
        Args:
            size: Number of values to generate
            low: Lower bound
            high: Upper bound
            
        Returns:
            Array of random values
        """
        log_low = np.log(low)
        log_high = np.log(high)
        return np.exp(self._rng.uniform(log_low, log_high, size))
    
    def generate(
        self,
        size: int,
        config: DisorderConfig
    ) -> np.ndarray:
        """
        Generate random values based on disorder configuration.
        
        Args:
            size: Number of values to generate
            config: Disorder configuration
            
        Returns:
            Array of random values
        """
        if config.disorder_type == DisorderType.BOX:
            return self.generate_box(size, config.center, config.width)
        elif config.disorder_type == DisorderType.GAUSSIAN:
            return self.generate_gaussian(size, config.center, config.width)
        elif config.disorder_type == DisorderType.BINARY:
            return self.generate_binary(size, (-config.center, config.center))
        elif config.disorder_type == DisorderType.LOGUNIFORM:
            low = config.center / (1 + config.width)
            high = config.center * (1 + config.width)
            return self.generate_loguniform(size, low, high)
        else:
            raise ValueError(f"Unknown disorder type: {config.disorder_type}")



@dataclass
class DisorderedSystemConfig:
    """Configuration for a disordered quantum system."""
    L: int  # System size
    J_config: DisorderConfig  # Coupling disorder
    h_config: DisorderConfig  # Transverse field disorder
    hz_config: Optional[DisorderConfig] = None  # Longitudinal field disorder
    periodic: bool = True  # Boundary conditions
    
    def __post_init__(self):
        if self.hz_config is None:
            # Default: no longitudinal field disorder
            self.hz_config = DisorderConfig(
                disorder_type=DisorderType.BOX,
                center=0.0,
                width=0.0
            )


class DisorderRealizationGenerator:
    """
    Generate complete disorder realizations for quantum systems.
    
    Creates consistent sets of random couplings and fields
    for use in disorder averaging.
    """
    
    def __init__(self, config: DisorderedSystemConfig, base_seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            config: System configuration
            base_seed: Base random seed (realizations use base_seed + i)
        """
        self.config = config
        self.base_seed = base_seed if base_seed is not None else 42
        self._generator = RandomCouplingGenerator()
    
    def generate_single(self, realization_index: int = 0) -> DisorderRealization:
        """
        Generate a single disorder realization.
        
        Args:
            realization_index: Index for seed generation
            
        Returns:
            DisorderRealization with all random parameters
        """
        seed = self.base_seed + realization_index
        self._generator.set_seed(seed)
        
        L = self.config.L
        n_bonds = L if self.config.periodic else L - 1
        
        # Generate couplings
        couplings = self._generator.generate(n_bonds, self.config.J_config)
        
        # Generate transverse fields
        transverse_fields = self._generator.generate(L, self.config.h_config)
        
        # Generate longitudinal fields
        longitudinal_fields = self._generator.generate(L, self.config.hz_config)
        
        return DisorderRealization(
            couplings=couplings,
            transverse_fields=transverse_fields,
            longitudinal_fields=longitudinal_fields,
            seed=seed
        )
    
    def generate_batch(self, n_realizations: int, start_index: int = 0) -> List[DisorderRealization]:
        """
        Generate multiple disorder realizations.
        
        Args:
            n_realizations: Number of realizations to generate
            start_index: Starting index for seed generation
            
        Returns:
            List of DisorderRealization objects
        """
        return [
            self.generate_single(start_index + i)
            for i in range(n_realizations)
        ]


@dataclass
class DisorderAverageResult:
    """Result of disorder averaging computation."""
    mean: float
    std: float
    stderr: float  # Standard error of the mean
    n_realizations: int
    values: np.ndarray  # Individual realization values
    
    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """95% confidence interval for the mean."""
        return (self.mean - 1.96 * self.stderr, self.mean + 1.96 * self.stderr)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'stderr': self.stderr,
            'n_realizations': self.n_realizations,
            'values': self.values.tolist()
        }


class DisorderAveragingFramework:
    """
    Framework for computing disorder-averaged quantities.
    
    Supports:
    - Sequential and parallel computation
    - Statistical analysis of results
    - Convergence monitoring
    """
    
    def __init__(
        self,
        config: DisorderedSystemConfig,
        base_seed: Optional[int] = None
    ):
        """
        Initialize framework.
        
        Args:
            config: System configuration
            base_seed: Base random seed
        """
        self.config = config
        self.base_seed = base_seed if base_seed is not None else 42
        self.realization_generator = DisorderRealizationGenerator(config, base_seed)
    
    def compute_single_realization(
        self,
        realization: DisorderRealization,
        compute_fn: Callable[[DisorderRealization], float]
    ) -> float:
        """
        Compute observable for a single realization.
        
        Args:
            realization: Disorder realization
            compute_fn: Function that computes observable from realization
            
        Returns:
            Observable value
        """
        return compute_fn(realization)
    
    def disorder_average(
        self,
        compute_fn: Callable[[DisorderRealization], float],
        n_realizations: int,
        parallel: bool = False,
        n_workers: Optional[int] = None
    ) -> DisorderAverageResult:
        """
        Compute disorder-averaged observable.
        
        Args:
            compute_fn: Function that computes observable from realization
            n_realizations: Number of disorder realizations
            parallel: Whether to use parallel computation
            n_workers: Number of parallel workers (default: CPU count)
            
        Returns:
            DisorderAverageResult with statistics
        """
        realizations = self.realization_generator.generate_batch(n_realizations)
        
        if parallel and n_realizations > 1:
            values = self._compute_parallel(realizations, compute_fn, n_workers)
        else:
            values = self._compute_sequential(realizations, compute_fn)
        
        values = np.array(values)
        
        return DisorderAverageResult(
            mean=np.mean(values),
            std=np.std(values, ddof=1) if len(values) > 1 else 0.0,
            stderr=np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
            n_realizations=n_realizations,
            values=values
        )
    
    def _compute_sequential(
        self,
        realizations: List[DisorderRealization],
        compute_fn: Callable[[DisorderRealization], float]
    ) -> List[float]:
        """Compute observables sequentially."""
        return [compute_fn(r) for r in realizations]
    
    def _compute_parallel(
        self,
        realizations: List[DisorderRealization],
        compute_fn: Callable[[DisorderRealization], float],
        n_workers: Optional[int] = None
    ) -> List[float]:
        """Compute observables in parallel using threads."""
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), len(realizations))
        
        # Use ThreadPoolExecutor for I/O-bound or GIL-releasing operations
        # For CPU-bound numpy operations, threads work well due to numpy releasing GIL
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(compute_fn, realizations))
        
        return results
    
    def disorder_average_multiple(
        self,
        compute_fns: Dict[str, Callable[[DisorderRealization], float]],
        n_realizations: int,
        parallel: bool = False,
        n_workers: Optional[int] = None
    ) -> Dict[str, DisorderAverageResult]:
        """
        Compute multiple disorder-averaged observables efficiently.
        
        Args:
            compute_fns: Dictionary of {name: compute_function}
            n_realizations: Number of disorder realizations
            parallel: Whether to use parallel computation
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary of {name: DisorderAverageResult}
        """
        realizations = self.realization_generator.generate_batch(n_realizations)
        
        # Compute all observables for each realization
        def compute_all(realization: DisorderRealization) -> Dict[str, float]:
            return {name: fn(realization) for name, fn in compute_fns.items()}
        
        if parallel and n_realizations > 1:
            if n_workers is None:
                n_workers = min(multiprocessing.cpu_count(), len(realizations))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                all_results = list(executor.map(compute_all, realizations))
        else:
            all_results = [compute_all(r) for r in realizations]
        
        # Reorganize results by observable
        results = {}
        for name in compute_fns.keys():
            values = np.array([r[name] for r in all_results])
            results[name] = DisorderAverageResult(
                mean=np.mean(values),
                std=np.std(values, ddof=1) if len(values) > 1 else 0.0,
                stderr=np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                n_realizations=n_realizations,
                values=values
            )
        
        return results



@dataclass
class StatisticalAnalysisResult:
    """Comprehensive statistical analysis of disorder-averaged data."""
    mean: float
    std: float
    stderr: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    skewness: float
    kurtosis: float
    n_samples: int
    histogram: Tuple[np.ndarray, np.ndarray]  # (counts, bin_edges)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'stderr': self.stderr,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'n_samples': self.n_samples,
            'histogram_counts': self.histogram[0].tolist(),
            'histogram_edges': self.histogram[1].tolist()
        }


class DisorderStatisticalAnalyzer:
    """
    Statistical analysis tools for disorder-averaged quantities.
    
    Provides:
    - Distribution analysis (mean, std, skewness, kurtosis)
    - Histogram generation
    - Convergence analysis
    - Bootstrap error estimation
    """
    
    @staticmethod
    def analyze_distribution(values: np.ndarray, n_bins: int = 50) -> StatisticalAnalysisResult:
        """
        Perform comprehensive statistical analysis of values.
        
        Args:
            values: Array of values from disorder realizations
            n_bins: Number of histogram bins
            
        Returns:
            StatisticalAnalysisResult with all statistics
        """
        n = len(values)
        if n == 0:
            raise ValueError("Cannot analyze empty array")
        
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        stderr = std / np.sqrt(n) if n > 1 else 0.0
        median = np.median(values)
        q25, q75 = np.percentile(values, [25, 75])
        
        # Skewness and kurtosis
        if n > 2 and std > 0:
            skewness = np.mean(((values - mean) / std) ** 3)
            kurtosis = np.mean(((values - mean) / std) ** 4) - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Histogram
        counts, bin_edges = np.histogram(values, bins=n_bins)
        
        return StatisticalAnalysisResult(
            mean=mean,
            std=std,
            stderr=stderr,
            median=median,
            q25=q25,
            q75=q75,
            skewness=skewness,
            kurtosis=kurtosis,
            n_samples=n,
            histogram=(counts, bin_edges)
        )
    
    @staticmethod
    def bootstrap_error(
        values: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float] = np.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: Optional[int] = None
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Estimate error using bootstrap resampling.
        
        Args:
            values: Array of values
            statistic_fn: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for interval
            seed: Random seed
            
        Returns:
            (estimate, std_error, confidence_interval)
        """
        rng = np.random.default_rng(seed)
        n = len(values)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        estimate = statistic_fn(values)
        std_error = np.std(bootstrap_stats, ddof=1)
        
        alpha = 1 - confidence
        ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return estimate, std_error, (ci_low, ci_high)
    
    @staticmethod
    def convergence_analysis(
        values: np.ndarray,
        window_sizes: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze convergence of disorder average with increasing samples.
        
        Args:
            values: Array of values (in order of computation)
            window_sizes: Sample sizes to analyze (default: logarithmic spacing)
            
        Returns:
            Dictionary with 'n_samples', 'running_mean', 'running_std', 'running_stderr'
        """
        n = len(values)
        
        if window_sizes is None:
            # Logarithmic spacing
            window_sizes = np.unique(np.logspace(0, np.log10(n), 20).astype(int))
            window_sizes = window_sizes[window_sizes <= n]
        
        running_mean = []
        running_std = []
        running_stderr = []
        
        for size in window_sizes:
            subset = values[:size]
            running_mean.append(np.mean(subset))
            if size > 1:
                std = np.std(subset, ddof=1)
                running_std.append(std)
                running_stderr.append(std / np.sqrt(size))
            else:
                running_std.append(0.0)
                running_stderr.append(0.0)
        
        return {
            'n_samples': np.array(window_sizes),
            'running_mean': np.array(running_mean),
            'running_std': np.array(running_std),
            'running_stderr': np.array(running_stderr)
        }
    
    @staticmethod
    def is_converged(
        values: np.ndarray,
        tolerance: float = 0.01,
        min_samples: int = 100
    ) -> Tuple[bool, float]:
        """
        Check if disorder average has converged.
        
        Convergence criterion: relative change in mean over last 20% of samples
        is less than tolerance.
        
        Args:
            values: Array of values
            tolerance: Relative tolerance for convergence
            min_samples: Minimum samples required
            
        Returns:
            (is_converged, relative_change)
        """
        n = len(values)
        
        if n < min_samples:
            return False, float('inf')
        
        # Compare mean of first 80% vs full sample
        n_80 = int(0.8 * n)
        mean_80 = np.mean(values[:n_80])
        mean_full = np.mean(values)
        
        if mean_80 == 0:
            relative_change = abs(mean_full - mean_80)
        else:
            relative_change = abs(mean_full - mean_80) / abs(mean_80)
        
        return relative_change < tolerance, relative_change
    
    @staticmethod
    def typical_vs_average(values: np.ndarray) -> Dict[str, float]:
        """
        Compare typical (geometric mean) vs arithmetic average.
        
        Important for disordered systems where rare events dominate.
        
        Args:
            values: Array of positive values
            
        Returns:
            Dictionary with 'arithmetic_mean', 'geometric_mean', 'ratio'
        """
        # Handle negative values by using log of absolute value
        positive_values = np.abs(values[values != 0])
        
        if len(positive_values) == 0:
            return {
                'arithmetic_mean': 0.0,
                'geometric_mean': 0.0,
                'ratio': 1.0
            }
        
        arithmetic_mean = np.mean(np.abs(values))
        geometric_mean = np.exp(np.mean(np.log(positive_values)))
        
        ratio = arithmetic_mean / geometric_mean if geometric_mean > 0 else float('inf')
        
        return {
            'arithmetic_mean': arithmetic_mean,
            'geometric_mean': geometric_mean,
            'ratio': ratio
        }
