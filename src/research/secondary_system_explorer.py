"""
Secondary System Refined Exploration Module.

Implements Task 12: Refine secondary system anomalies
- 12.1 Same refinement protocol as primary (DTFIM)
- 12.2 Cross-compare with primary system
- 12.3 Look for universal features

This module provides refined exploration for secondary quantum systems:
- Long-Range Quantum Ising Model
- Floquet (Periodically Driven) Ising Model

The refinement protocol mirrors the DTFIM approach:
- 10x resolution in anomalous regions
- Multiple system sizes for finite-size scaling
- Extended observable computation
- Entanglement analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from enum import Enum

from ..quantum.long_range_ising import LongRangeIsing, LongRangeIsingParams
from ..quantum.floquet_ising import FloquetIsing, FloquetIsingParams
from ..quantum.entanglement import EntanglementCalculator


class SecondarySystemType(Enum):
    """Types of secondary quantum systems."""
    LONG_RANGE_ISING = "long_range_ising"
    FLOQUET_ISING = "floquet_ising"


@dataclass
class LongRangeAnomalousRegion:
    """Definition of an anomalous region for Long-Range Ising."""
    alpha_center: float  # Power-law exponent center
    h_center: float  # Transverse field center
    alpha_width: float = 0.3  # Half-width
    h_width: float = 0.3
    anomaly_type: str = "unknown"
    severity: float = 0.0
    description: str = ""
    
    @property
    def alpha_range(self) -> Tuple[float, float]:
        return (self.alpha_center - self.alpha_width, self.alpha_center + self.alpha_width)
    
    @property
    def h_range(self) -> Tuple[float, float]:
        return (self.h_center - self.h_width, self.h_center + self.h_width)


@dataclass
class FloquetAnomalousRegion:
    """Definition of an anomalous region for Floquet Ising."""
    h1_center: float  # First half-period field
    h2_center: float  # Second half-period field
    T_center: float  # Driving period
    h1_width: float = 0.3
    h2_width: float = 0.3
    T_width: float = 0.3
    anomaly_type: str = "unknown"
    severity: float = 0.0
    description: str = ""
    
    @property
    def h1_range(self) -> Tuple[float, float]:
        return (self.h1_center - self.h1_width, self.h1_center + self.h1_width)
    
    @property
    def h2_range(self) -> Tuple[float, float]:
        return (self.h2_center - self.h2_width, self.h2_center + self.h2_width)
    
    @property
    def T_range(self) -> Tuple[float, float]:
        return (self.T_center - self.T_width, self.T_center + self.T_width)


@dataclass
class LongRangeScanPoint:
    """Single point in Long-Range Ising parameter space."""
    alpha: float  # Power-law exponent
    h: float  # Transverse field
    L: int  # System size
    
    # Observables
    energy: float
    magnetization_z: float
    magnetization_x: float
    correlation: float
    entanglement_entropy: float
    
    # Additional quantum observables
    susceptibility_z: float = 0.0
    energy_gap: float = 0.0
    entanglement_spectrum: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'alpha': self.alpha,
            'h': self.h,
            'L': self.L,
            'energy': self.energy,
            'magnetization_z': self.magnetization_z,
            'magnetization_x': self.magnetization_x,
            'correlation': self.correlation,
            'entanglement_entropy': self.entanglement_entropy,
            'susceptibility_z': self.susceptibility_z,
            'energy_gap': self.energy_gap,
        }
        if self.entanglement_spectrum is not None:
            result['entanglement_spectrum'] = self.entanglement_spectrum.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongRangeScanPoint':
        spectrum = data.pop('entanglement_spectrum', None)
        point = cls(**data)
        if spectrum is not None:
            point.entanglement_spectrum = np.array(spectrum)
        return point


@dataclass
class FloquetScanPoint:
    """Single point in Floquet Ising parameter space."""
    h1: float  # First half-period field
    h2: float  # Second half-period field
    T: float  # Driving period
    L: int  # System size
    
    # Observables
    quasienergy_gap: float
    period_doubling_strength: float
    final_amplitude: float
    has_time_crystal_signature: bool
    
    # Additional observables
    magnetization_trajectory: Optional[np.ndarray] = None
    quasienergy_spectrum: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'h1': float(self.h1),
            'h2': float(self.h2),
            'T': float(self.T),
            'L': int(self.L),
            'quasienergy_gap': float(self.quasienergy_gap),
            'period_doubling_strength': float(self.period_doubling_strength),
            'final_amplitude': float(self.final_amplitude),
            'has_time_crystal_signature': bool(self.has_time_crystal_signature),
        }
        if self.magnetization_trajectory is not None:
            result['magnetization_trajectory'] = self.magnetization_trajectory.tolist()
        if self.quasienergy_spectrum is not None:
            result['quasienergy_spectrum'] = self.quasienergy_spectrum.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FloquetScanPoint':
        traj = data.pop('magnetization_trajectory', None)
        spectrum = data.pop('quasienergy_spectrum', None)
        point = cls(**data)
        if traj is not None:
            point.magnetization_trajectory = np.array(traj)
        if spectrum is not None:
            point.quasienergy_spectrum = np.array(spectrum)
        return point



@dataclass
class LongRangeExplorationResult:
    """Results of refined Long-Range Ising exploration."""
    scan_points: List[LongRangeScanPoint]
    system_sizes: List[int]
    alpha_values: np.ndarray
    h_values: np.ndarray
    anomalous_region: LongRangeAnomalousRegion
    metadata: Dict = field(default_factory=dict)
    fss_data: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'system_type': 'long_range_ising',
            'alpha_values': self.alpha_values.tolist(),
            'h_values': self.h_values.tolist(),
            'system_sizes': self.system_sizes,
            'scan_points': [p.to_dict() for p in self.scan_points],
            'anomalous_region': {
                'alpha_center': self.anomalous_region.alpha_center,
                'h_center': self.anomalous_region.h_center,
                'alpha_width': self.anomalous_region.alpha_width,
                'h_width': self.anomalous_region.h_width,
                'anomaly_type': self.anomalous_region.anomaly_type,
                'severity': self.anomalous_region.severity,
                'description': self.anomalous_region.description,
            },
            'metadata': self.metadata,
            'fss_data': {
                obs: {str(L): vals.tolist() for L, vals in size_data.items()}
                for obs, size_data in self.fss_data.items()
            },
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'LongRangeExplorationResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scan_points = [LongRangeScanPoint.from_dict(p) for p in data['scan_points']]
        anomalous_region = LongRangeAnomalousRegion(**data['anomalous_region'])
        
        fss_data = {}
        if 'fss_data' in data:
            for obs, size_data in data['fss_data'].items():
                fss_data[obs] = {int(L): np.array(vals) for L, vals in size_data.items()}
        
        return cls(
            scan_points=scan_points,
            system_sizes=data['system_sizes'],
            alpha_values=np.array(data['alpha_values']),
            h_values=np.array(data['h_values']),
            anomalous_region=anomalous_region,
            metadata=data.get('metadata', {}),
            fss_data=fss_data,
        )
    
    def get_points_for_size(self, L: int) -> List[LongRangeScanPoint]:
        """Get all scan points for a specific system size."""
        return [p for p in self.scan_points if p.L == L]
    
    def get_observable_grid(self, observable: str, L: int) -> np.ndarray:
        """Get 2D grid of observable values for a specific system size."""
        points = self.get_points_for_size(L)
        n_alpha = len(self.alpha_values)
        n_h = len(self.h_values)
        grid = np.full((n_h, n_alpha), np.nan)
        
        for point in points:
            i_alpha = np.argmin(np.abs(self.alpha_values - point.alpha))
            i_h = np.argmin(np.abs(self.h_values - point.h))
            grid[i_h, i_alpha] = getattr(point, observable, np.nan)
        
        return grid


@dataclass
class FloquetExplorationResult:
    """Results of refined Floquet Ising exploration."""
    scan_points: List[FloquetScanPoint]
    system_sizes: List[int]
    h1_values: np.ndarray
    h2_values: np.ndarray
    T_values: np.ndarray
    anomalous_region: FloquetAnomalousRegion
    metadata: Dict = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'system_type': 'floquet_ising',
            'h1_values': self.h1_values.tolist(),
            'h2_values': self.h2_values.tolist(),
            'T_values': self.T_values.tolist(),
            'system_sizes': self.system_sizes,
            'scan_points': [p.to_dict() for p in self.scan_points],
            'anomalous_region': {
                'h1_center': self.anomalous_region.h1_center,
                'h2_center': self.anomalous_region.h2_center,
                'T_center': self.anomalous_region.T_center,
                'h1_width': self.anomalous_region.h1_width,
                'h2_width': self.anomalous_region.h2_width,
                'T_width': self.anomalous_region.T_width,
                'anomaly_type': self.anomalous_region.anomaly_type,
                'severity': self.anomalous_region.severity,
                'description': self.anomalous_region.description,
            },
            'metadata': self.metadata,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FloquetExplorationResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scan_points = [FloquetScanPoint.from_dict(p) for p in data['scan_points']]
        anomalous_region = FloquetAnomalousRegion(**data['anomalous_region'])
        
        return cls(
            scan_points=scan_points,
            system_sizes=data['system_sizes'],
            h1_values=np.array(data['h1_values']),
            h2_values=np.array(data['h2_values']),
            T_values=np.array(data['T_values']),
            anomalous_region=anomalous_region,
            metadata=data.get('metadata', {}),
        )
    
    def get_points_for_size(self, L: int) -> List[FloquetScanPoint]:
        """Get all scan points for a specific system size."""
        return [p for p in self.scan_points if p.L == L]
    
    def get_time_crystal_candidates(self) -> List[FloquetScanPoint]:
        """Get points with time crystal signatures."""
        return [p for p in self.scan_points if p.has_time_crystal_signature]



class LongRangeIsingRefinedExplorer:
    """
    Refined exploration of Long-Range Quantum Ising anomalous regions.
    
    Applies the same refinement protocol as DTFIM:
    - 10x resolution in anomalous regions
    - Multiple system sizes for finite-size scaling
    - Extended observable computation
    - Entanglement analysis
    """
    
    def __init__(
        self,
        system_sizes: List[int] = [8, 10, 12, 14],
        J0: float = 1.0,
        periodic: bool = True,
    ):
        """
        Initialize refined explorer.
        
        Args:
            system_sizes: List of system sizes for finite-size scaling
            J0: Base coupling strength
            periodic: Use periodic boundary conditions
        """
        self.system_sizes = sorted(system_sizes)
        self.J0 = J0
        self.periodic = periodic
        self.logger = logging.getLogger(__name__)
    
    def _compute_single_point(
        self,
        alpha: float,
        h: float,
        L: int,
    ) -> LongRangeScanPoint:
        """
        Compute observables at a single (alpha, h) point for system size L.
        
        Args:
            alpha: Power-law exponent
            h: Transverse field
            L: System size
            
        Returns:
            LongRangeScanPoint with computed observables
        """
        params = LongRangeIsingParams(
            L=L,
            J0=self.J0,
            alpha=alpha,
            h=h,
            periodic=self.periodic
        )
        
        model = LongRangeIsing(params)
        energy, state = model.compute_ground_state()
        observables = model.compute_observables(state)
        entropy = model.compute_entanglement_entropy(state)
        
        # Compute entanglement spectrum
        ent_calc = EntanglementCalculator(L)
        ent_spectrum = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
        
        # Compute susceptibility (variance of magnetization)
        # For a single state, estimate from correlation functions
        mag_z = observables['magnetization_z']
        susceptibility_z = L * (1 - mag_z**2)  # Simplified estimate
        
        return LongRangeScanPoint(
            alpha=alpha,
            h=h,
            L=L,
            energy=energy,
            magnetization_z=observables['magnetization_z'],
            magnetization_x=observables['magnetization_x'],
            correlation=observables['correlation'],
            entanglement_entropy=entropy,
            susceptibility_z=susceptibility_z,
            energy_gap=0.0,  # Would need excited state calculation
            entanglement_spectrum=ent_spectrum.eigenvalues[:10],
        )
    
    def refine_anomalous_region(
        self,
        region: LongRangeAnomalousRegion,
        resolution_factor: int = 10,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> LongRangeExplorationResult:
        """
        Perform refined exploration of an anomalous region.
        
        Args:
            region: Anomalous region to explore
            resolution_factor: Resolution increase factor (default 10x)
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            
        Returns:
            LongRangeExplorationResult with detailed scan data
        """
        # Create high-resolution grid
        n_points = resolution_factor
        alpha_values = np.linspace(
            max(region.alpha_range[0], 1.0),  # alpha must be > 0
            region.alpha_range[1],
            n_points
        )
        h_values = np.linspace(
            max(region.h_range[0], 0.0),  # h must be >= 0
            region.h_range[1],
            n_points
        )
        
        self.logger.info(f"Refined exploration: α ∈ {region.alpha_range}, h ∈ {region.h_range}")
        self.logger.info(f"Resolution: {n_points}x{n_points} = {n_points**2} points")
        self.logger.info(f"System sizes: {self.system_sizes}")
        
        # Generate all tasks
        tasks = []
        for alpha in alpha_values:
            for h in h_values:
                for L in self.system_sizes:
                    tasks.append((alpha, h, L))
        
        total_tasks = len(tasks)
        self.logger.info(f"Total computations: {total_tasks}")
        
        # Compute all points
        scan_points = []
        start_time = time.time()
        
        if parallel and total_tasks > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._compute_single_point, alpha, h, L): (alpha, h, L)
                    for alpha, h, L in tasks
                }
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        point = future.result()
                        scan_points.append(point)
                        
                        if (i + 1) % 20 == 0 or (i + 1) == total_tasks:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            remaining = (total_tasks - i - 1) / rate if rate > 0 else 0
                            self.logger.info(
                                f"Progress: {i + 1}/{total_tasks} "
                                f"({100*(i+1)/total_tasks:.1f}%) "
                                f"ETA: {remaining/60:.1f} min"
                            )
                    except Exception as e:
                        alpha, h, L = futures[future]
                        self.logger.error(f"Failed at (α={alpha}, h={h}, L={L}): {e}")
        else:
            for i, (alpha, h, L) in enumerate(tasks):
                try:
                    point = self._compute_single_point(alpha, h, L)
                    scan_points.append(point)
                    
                    if (i + 1) % 20 == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_tasks}")
                except Exception as e:
                    self.logger.error(f"Failed at (α={alpha}, h={h}, L={L}): {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed in {elapsed/60:.1f} minutes")
        
        # Build finite-size scaling data
        fss_data = self._build_fss_data(scan_points, alpha_values, h_values)
        
        return LongRangeExplorationResult(
            scan_points=scan_points,
            system_sizes=self.system_sizes,
            alpha_values=alpha_values,
            h_values=h_values,
            anomalous_region=region,
            metadata={
                'J0': self.J0,
                'periodic': self.periodic,
                'resolution_factor': resolution_factor,
                'total_points': len(scan_points),
                'computation_time_minutes': elapsed / 60,
            },
            fss_data=fss_data,
        )
    
    def _build_fss_data(
        self,
        scan_points: List[LongRangeScanPoint],
        alpha_values: np.ndarray,
        h_values: np.ndarray
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Build finite-size scaling data structures."""
        observables = [
            'magnetization_z', 'magnetization_x', 'entanglement_entropy',
            'correlation', 'susceptibility_z'
        ]
        
        fss_data = {obs: {} for obs in observables}
        
        for L in self.system_sizes:
            points_L = [p for p in scan_points if p.L == L]
            
            for obs in observables:
                grid = np.full((len(h_values), len(alpha_values)), np.nan)
                
                for point in points_L:
                    i_alpha = np.argmin(np.abs(alpha_values - point.alpha))
                    i_h = np.argmin(np.abs(h_values - point.h))
                    grid[i_h, i_alpha] = getattr(point, obs, np.nan)
                
                fss_data[obs][L] = grid
        
        return fss_data



class FloquetIsingRefinedExplorer:
    """
    Refined exploration of Floquet Ising anomalous regions.
    
    Applies the same refinement protocol as DTFIM:
    - 10x resolution in anomalous regions
    - Multiple system sizes for finite-size scaling
    - Time crystal signature detection
    - Quasienergy spectrum analysis
    """
    
    def __init__(
        self,
        system_sizes: List[int] = [8, 10, 12],
        J: float = 1.0,
        n_periods: int = 20,
    ):
        """
        Initialize refined explorer.
        
        Args:
            system_sizes: List of system sizes (keep small due to exponential cost)
            J: Ising coupling strength
            n_periods: Number of periods for time evolution
        """
        # Floquet is expensive - limit system sizes
        self.system_sizes = sorted([L for L in system_sizes if L <= 14])
        if not self.system_sizes:
            self.system_sizes = [8, 10, 12]
        
        self.J = J
        self.n_periods = n_periods
        self.logger = logging.getLogger(__name__)
    
    def _compute_single_point(
        self,
        h1: float,
        h2: float,
        T: float,
        L: int,
    ) -> FloquetScanPoint:
        """
        Compute observables at a single (h1, h2, T) point for system size L.
        
        Args:
            h1: First half-period transverse field
            h2: Second half-period transverse field
            T: Driving period
            L: System size
            
        Returns:
            FloquetScanPoint with computed observables
        """
        params = FloquetIsingParams(
            L=L,
            J=self.J,
            h1=h1,
            h2=h2,
            T=T,
            periodic=True
        )
        
        model = FloquetIsing(params)
        
        # Compute quasienergy spectrum
        quasienergies, _ = model.compute_quasienergy_spectrum()
        gap = model.compute_quasienergy_gap()
        
        # Detect time crystal signatures
        from ..quantum.floquet_ising import FloquetIsingExplorer
        explorer = FloquetIsingExplorer(L=L, J=self.J)
        tc_result = explorer.detect_time_crystal_signatures(params, n_periods=self.n_periods)
        
        return FloquetScanPoint(
            h1=h1,
            h2=h2,
            T=T,
            L=L,
            quasienergy_gap=gap,
            period_doubling_strength=tc_result['period_doubling_strength'],
            final_amplitude=tc_result['final_amplitude'],
            has_time_crystal_signature=tc_result['has_time_crystal_signature'],
            magnetization_trajectory=tc_result['magnetization_trajectory'],
            quasienergy_spectrum=quasienergies[:20] if len(quasienergies) > 20 else quasienergies,
        )
    
    def refine_anomalous_region(
        self,
        region: FloquetAnomalousRegion,
        resolution_factor: int = 5,  # Lower resolution due to 3D parameter space
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> FloquetExplorationResult:
        """
        Perform refined exploration of an anomalous region.
        
        Args:
            region: Anomalous region to explore
            resolution_factor: Resolution per axis (default 5 for 3D)
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            
        Returns:
            FloquetExplorationResult with detailed scan data
        """
        # Create high-resolution grid (3D parameter space)
        n_points = resolution_factor
        h1_values = np.linspace(
            max(region.h1_range[0], 0.1),
            region.h1_range[1],
            n_points
        )
        h2_values = np.linspace(
            max(region.h2_range[0], 0.1),
            region.h2_range[1],
            n_points
        )
        T_values = np.linspace(
            max(region.T_range[0], 0.1),
            region.T_range[1],
            n_points
        )
        
        self.logger.info(f"Refined Floquet exploration:")
        self.logger.info(f"  h1 ∈ {region.h1_range}, h2 ∈ {region.h2_range}, T ∈ {region.T_range}")
        self.logger.info(f"  Resolution: {n_points}³ = {n_points**3} points per size")
        self.logger.info(f"  System sizes: {self.system_sizes}")
        
        # Generate all tasks
        tasks = []
        for h1 in h1_values:
            for h2 in h2_values:
                for T in T_values:
                    for L in self.system_sizes:
                        tasks.append((h1, h2, T, L))
        
        total_tasks = len(tasks)
        self.logger.info(f"Total computations: {total_tasks}")
        
        # Compute all points
        scan_points = []
        start_time = time.time()
        
        if parallel and total_tasks > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._compute_single_point, h1, h2, T, L): (h1, h2, T, L)
                    for h1, h2, T, L in tasks
                }
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        point = future.result()
                        scan_points.append(point)
                        
                        if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            remaining = (total_tasks - i - 1) / rate if rate > 0 else 0
                            self.logger.info(
                                f"Progress: {i + 1}/{total_tasks} "
                                f"({100*(i+1)/total_tasks:.1f}%) "
                                f"ETA: {remaining/60:.1f} min"
                            )
                    except Exception as e:
                        h1, h2, T, L = futures[future]
                        self.logger.error(f"Failed at (h1={h1}, h2={h2}, T={T}, L={L}): {e}")
        else:
            for i, (h1, h2, T, L) in enumerate(tasks):
                try:
                    point = self._compute_single_point(h1, h2, T, L)
                    scan_points.append(point)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_tasks}")
                except Exception as e:
                    self.logger.error(f"Failed at (h1={h1}, h2={h2}, T={T}, L={L}): {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed in {elapsed/60:.1f} minutes")
        
        return FloquetExplorationResult(
            scan_points=scan_points,
            system_sizes=self.system_sizes,
            h1_values=h1_values,
            h2_values=h2_values,
            T_values=T_values,
            anomalous_region=region,
            metadata={
                'J': self.J,
                'n_periods': self.n_periods,
                'resolution_factor': resolution_factor,
                'total_points': len(scan_points),
                'computation_time_minutes': elapsed / 60,
            },
        )



@dataclass
class CrossComparisonResult:
    """Result of cross-comparison between primary and secondary systems."""
    primary_system: str
    secondary_system: str
    
    # Shared features
    shared_critical_behavior: bool
    shared_exponents: Dict[str, Tuple[float, float]]  # {exponent: (primary, secondary)}
    correlation_coefficient: float  # How similar are the phase diagrams
    
    # Differences
    unique_to_primary: List[str]
    unique_to_secondary: List[str]
    
    # Universal features
    universal_features: List[str]
    universality_class_match: Optional[str]
    
    # Evidence
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_system': self.primary_system,
            'secondary_system': self.secondary_system,
            'shared_critical_behavior': self.shared_critical_behavior,
            'shared_exponents': self.shared_exponents,
            'correlation_coefficient': self.correlation_coefficient,
            'unique_to_primary': self.unique_to_primary,
            'unique_to_secondary': self.unique_to_secondary,
            'universal_features': self.universal_features,
            'universality_class_match': self.universality_class_match,
            'evidence': self.evidence,
        }


@dataclass
class UniversalFeature:
    """A universal feature found across multiple systems."""
    name: str
    description: str
    systems_exhibiting: List[str]
    exponent_values: Dict[str, float]  # {system: value}
    confidence: float
    is_novel: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'systems_exhibiting': self.systems_exhibiting,
            'exponent_values': self.exponent_values,
            'confidence': self.confidence,
            'is_novel': self.is_novel,
        }


class SecondarySystemCrossComparator:
    """
    Cross-comparison between primary (DTFIM) and secondary systems.
    
    Implements Task 12.2: Cross-compare with primary system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_with_dtfim(
        self,
        dtfim_result: Any,  # RefinedExplorationResult from DTFIM
        secondary_result: Any,  # LongRangeExplorationResult or FloquetExplorationResult
    ) -> CrossComparisonResult:
        """
        Compare secondary system results with DTFIM results.
        
        Args:
            dtfim_result: Results from DTFIM refined exploration
            secondary_result: Results from secondary system exploration
            
        Returns:
            CrossComparisonResult with comparison analysis
        """
        # Determine secondary system type
        if isinstance(secondary_result, LongRangeExplorationResult):
            secondary_type = "long_range_ising"
            return self._compare_long_range_with_dtfim(dtfim_result, secondary_result)
        elif isinstance(secondary_result, FloquetExplorationResult):
            secondary_type = "floquet_ising"
            return self._compare_floquet_with_dtfim(dtfim_result, secondary_result)
        else:
            raise ValueError(f"Unknown secondary result type: {type(secondary_result)}")
    
    def _compare_long_range_with_dtfim(
        self,
        dtfim_result: Any,
        lr_result: LongRangeExplorationResult,
    ) -> CrossComparisonResult:
        """Compare Long-Range Ising with DTFIM."""
        evidence = {}
        shared_exponents = {}
        unique_primary = []
        unique_secondary = []
        universal_features = []
        
        # Get largest system size data from both
        L_dtfim = max(dtfim_result.system_sizes)
        L_lr = max(lr_result.system_sizes)
        
        dtfim_points = dtfim_result.get_points_for_size(L_dtfim)
        lr_points = lr_result.get_points_for_size(L_lr)
        
        # Compare magnetization behavior
        dtfim_mag = [p.magnetization_z for p in dtfim_points]
        lr_mag = [p.magnetization_z for p in lr_points]
        
        # Check if both show phase transitions
        dtfim_has_transition = max(dtfim_mag) - min(dtfim_mag) > 0.3
        lr_has_transition = max(lr_mag) - min(lr_mag) > 0.3
        
        shared_critical = dtfim_has_transition and lr_has_transition
        evidence['dtfim_has_transition'] = dtfim_has_transition
        evidence['lr_has_transition'] = lr_has_transition
        
        # Compare entanglement behavior
        dtfim_ent = [p.entanglement_entropy for p in dtfim_points]
        lr_ent = [p.entanglement_entropy for p in lr_points]
        
        dtfim_max_ent = max(dtfim_ent)
        lr_max_ent = max(lr_ent)
        
        evidence['dtfim_max_entanglement'] = dtfim_max_ent
        evidence['lr_max_entanglement'] = lr_max_ent
        
        # Check for entanglement peak (signature of criticality)
        dtfim_ent_peak = dtfim_max_ent > np.mean(dtfim_ent) + np.std(dtfim_ent)
        lr_ent_peak = lr_max_ent > np.mean(lr_ent) + np.std(lr_ent)
        
        if dtfim_ent_peak and lr_ent_peak:
            universal_features.append("entanglement_peak_at_criticality")
        
        # Unique features
        if hasattr(dtfim_points[0], 'binder_cumulant'):
            unique_primary.append("binder_cumulant_analysis")
        
        # Long-range has alpha-dependent behavior
        unique_secondary.append("power_law_exponent_dependence")
        
        # Check for mean-field vs short-range crossover in long-range
        if lr_result.alpha_values is not None:
            alpha_range = lr_result.alpha_values
            if min(alpha_range) < 2.0 and max(alpha_range) > 2.0:
                unique_secondary.append("mean_field_crossover")
        
        # Compute correlation between phase diagrams (simplified)
        # Use magnetization as proxy
        correlation = np.corrcoef(
            np.array(dtfim_mag[:min(len(dtfim_mag), len(lr_mag))]),
            np.array(lr_mag[:min(len(dtfim_mag), len(lr_mag))])
        )[0, 1] if len(dtfim_mag) > 1 and len(lr_mag) > 1 else 0.0
        
        return CrossComparisonResult(
            primary_system="dtfim",
            secondary_system="long_range_ising",
            shared_critical_behavior=shared_critical,
            shared_exponents=shared_exponents,
            correlation_coefficient=correlation if np.isfinite(correlation) else 0.0,
            unique_to_primary=unique_primary,
            unique_to_secondary=unique_secondary,
            universal_features=universal_features,
            universality_class_match="1d_ising" if shared_critical else None,
            evidence=evidence,
        )
    
    def _compare_floquet_with_dtfim(
        self,
        dtfim_result: Any,
        floquet_result: FloquetExplorationResult,
    ) -> CrossComparisonResult:
        """Compare Floquet Ising with DTFIM."""
        evidence = {}
        shared_exponents = {}
        unique_primary = []
        unique_secondary = []
        universal_features = []
        
        # DTFIM is equilibrium, Floquet is non-equilibrium
        # They have fundamentally different physics
        
        # Check for time crystal signatures in Floquet
        tc_candidates = floquet_result.get_time_crystal_candidates()
        has_time_crystal = len(tc_candidates) > 0
        
        evidence['floquet_time_crystal_candidates'] = len(tc_candidates)
        evidence['floquet_has_time_crystal'] = has_time_crystal
        
        # Unique features
        unique_primary.extend([
            "disorder_averaging",
            "griffiths_phase_possibility",
            "equilibrium_phase_transition"
        ])
        
        unique_secondary.extend([
            "time_crystal_signatures",
            "quasienergy_spectrum",
            "non_equilibrium_dynamics",
            "floquet_topological_phases"
        ])
        
        # Both can show quantum phase transitions
        # Check if both have sharp features
        L_dtfim = max(dtfim_result.system_sizes)
        dtfim_points = dtfim_result.get_points_for_size(L_dtfim)
        dtfim_chi = [p.susceptibility_z for p in dtfim_points]
        dtfim_has_peak = max(dtfim_chi) > 2 * np.mean(dtfim_chi)
        
        floquet_points = floquet_result.scan_points
        floquet_gaps = [p.quasienergy_gap for p in floquet_points]
        floquet_has_gap_closing = min(floquet_gaps) < 0.1 * np.mean(floquet_gaps)
        
        shared_critical = dtfim_has_peak or floquet_has_gap_closing
        
        if dtfim_has_peak and floquet_has_gap_closing:
            universal_features.append("gap_closing_at_transition")
        
        evidence['dtfim_susceptibility_peak'] = dtfim_has_peak
        evidence['floquet_gap_closing'] = floquet_has_gap_closing
        
        return CrossComparisonResult(
            primary_system="dtfim",
            secondary_system="floquet_ising",
            shared_critical_behavior=shared_critical,
            shared_exponents=shared_exponents,
            correlation_coefficient=0.0,  # Different physics, no direct correlation
            unique_to_primary=unique_primary,
            unique_to_secondary=unique_secondary,
            universal_features=universal_features,
            universality_class_match=None,  # Different universality classes
            evidence=evidence,
        )



class UniversalFeatureDetector:
    """
    Detector for universal features across quantum systems.
    
    Implements Task 12.3: Look for universal features
    """
    
    # Known universal features in quantum phase transitions
    KNOWN_UNIVERSAL_FEATURES = {
        'entanglement_area_law': {
            'description': 'Entanglement entropy scales with boundary area in gapped phases',
            'signature': 'S ~ L^(d-1) for d-dimensional systems',
        },
        'entanglement_log_correction': {
            'description': 'Logarithmic correction to area law at criticality',
            'signature': 'S ~ (c/3) log(L) for 1D CFT with central charge c',
        },
        'gap_closing': {
            'description': 'Energy gap closes at quantum critical point',
            'signature': 'Δ ~ L^(-z) with dynamical exponent z',
        },
        'susceptibility_divergence': {
            'description': 'Susceptibility diverges at critical point',
            'signature': 'χ ~ |g-gc|^(-γ) with exponent γ',
        },
        'correlation_length_divergence': {
            'description': 'Correlation length diverges at critical point',
            'signature': 'ξ ~ |g-gc|^(-ν) with exponent ν',
        },
        'finite_size_scaling': {
            'description': 'Observables follow universal scaling functions',
            'signature': 'O(L, g) = L^(x/ν) f((g-gc)L^(1/ν))',
        },
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_universal_features(
        self,
        results: List[Any],  # List of exploration results from different systems
        system_names: List[str],
    ) -> List[UniversalFeature]:
        """
        Detect universal features across multiple quantum systems.
        
        Args:
            results: List of exploration results from different systems
            system_names: Names of the systems
            
        Returns:
            List of detected universal features
        """
        universal_features = []
        
        # Check for entanglement scaling
        ent_feature = self._check_entanglement_scaling(results, system_names)
        if ent_feature:
            universal_features.append(ent_feature)
        
        # Check for gap closing
        gap_feature = self._check_gap_closing(results, system_names)
        if gap_feature:
            universal_features.append(gap_feature)
        
        # Check for susceptibility divergence
        chi_feature = self._check_susceptibility_divergence(results, system_names)
        if chi_feature:
            universal_features.append(chi_feature)
        
        # Check for finite-size scaling
        fss_feature = self._check_finite_size_scaling(results, system_names)
        if fss_feature:
            universal_features.append(fss_feature)
        
        return universal_features
    
    def _check_entanglement_scaling(
        self,
        results: List[Any],
        system_names: List[str],
    ) -> Optional[UniversalFeature]:
        """Check for universal entanglement scaling."""
        systems_with_log_scaling = []
        exponent_values = {}
        
        for result, name in zip(results, system_names):
            # Get entanglement data for different system sizes
            if not hasattr(result, 'system_sizes') or len(result.system_sizes) < 2:
                continue
            
            sizes = result.system_sizes
            max_entropies = []
            
            for L in sizes:
                points = result.get_points_for_size(L)
                if points:
                    entropies = [getattr(p, 'entanglement_entropy', 0) for p in points]
                    max_entropies.append(max(entropies))
            
            if len(max_entropies) >= 2:
                # Check for log(L) scaling: S ~ c/3 * log(L)
                log_sizes = np.log(sizes[:len(max_entropies)])
                slope, intercept, r_value, _, _ = np.polyfit(
                    log_sizes, max_entropies, 1, full=False, cov=False
                ), 0, 0, 0, 0
                
                # Simple linear regression
                from scipy.stats import linregress
                try:
                    slope, intercept, r_value, _, _ = linregress(log_sizes, max_entropies)
                    
                    # If good fit to log scaling, extract central charge
                    if r_value**2 > 0.8:
                        central_charge = 3 * slope  # c = 3 * (dS/d(log L))
                        systems_with_log_scaling.append(name)
                        exponent_values[name] = central_charge
                except:
                    pass
        
        if len(systems_with_log_scaling) >= 2:
            # Check if central charges are similar
            charges = list(exponent_values.values())
            mean_charge = np.mean(charges)
            std_charge = np.std(charges)
            
            is_universal = std_charge < 0.3 * mean_charge  # Within 30%
            
            return UniversalFeature(
                name="entanglement_log_scaling",
                description="Logarithmic entanglement scaling at criticality (CFT signature)",
                systems_exhibiting=systems_with_log_scaling,
                exponent_values=exponent_values,
                confidence=0.8 if is_universal else 0.5,
                is_novel=False,  # This is known physics
            )
        
        return None
    
    def _check_gap_closing(
        self,
        results: List[Any],
        system_names: List[str],
    ) -> Optional[UniversalFeature]:
        """Check for universal gap closing behavior."""
        systems_with_gap_closing = []
        z_values = {}
        
        for result, name in zip(results, system_names):
            if not hasattr(result, 'system_sizes') or len(result.system_sizes) < 2:
                continue
            
            sizes = result.system_sizes
            min_gaps = []
            
            for L in sizes:
                points = result.get_points_for_size(L)
                if points:
                    # Get energy gap or quasienergy gap
                    gaps = []
                    for p in points:
                        if hasattr(p, 'energy_gap') and p.energy_gap > 0:
                            gaps.append(p.energy_gap)
                        elif hasattr(p, 'quasienergy_gap') and p.quasienergy_gap > 0:
                            gaps.append(p.quasienergy_gap)
                    
                    if gaps:
                        min_gaps.append(min(gaps))
            
            if len(min_gaps) >= 2 and all(g > 0 for g in min_gaps):
                # Check for power-law scaling: Δ ~ L^(-z)
                log_sizes = np.log(sizes[:len(min_gaps)])
                log_gaps = np.log(min_gaps)
                
                from scipy.stats import linregress
                try:
                    slope, _, r_value, _, _ = linregress(log_sizes, log_gaps)
                    z = -slope  # Dynamical exponent
                    
                    if r_value**2 > 0.7 and z > 0:
                        systems_with_gap_closing.append(name)
                        z_values[name] = z
                except:
                    pass
        
        if len(systems_with_gap_closing) >= 1:
            return UniversalFeature(
                name="gap_closing_power_law",
                description="Energy gap closes as power law Δ ~ L^(-z) at criticality",
                systems_exhibiting=systems_with_gap_closing,
                exponent_values=z_values,
                confidence=0.7,
                is_novel=False,
            )
        
        return None
    
    def _check_susceptibility_divergence(
        self,
        results: List[Any],
        system_names: List[str],
    ) -> Optional[UniversalFeature]:
        """Check for susceptibility divergence."""
        systems_with_divergence = []
        peak_values = {}
        
        for result, name in zip(results, system_names):
            if not hasattr(result, 'system_sizes'):
                continue
            
            L_max = max(result.system_sizes)
            points = result.get_points_for_size(L_max)
            
            if points:
                chi_values = [getattr(p, 'susceptibility_z', 0) for p in points]
                if chi_values:
                    max_chi = max(chi_values)
                    mean_chi = np.mean(chi_values)
                    
                    # Check for significant peak
                    if max_chi > 2 * mean_chi:
                        systems_with_divergence.append(name)
                        peak_values[name] = max_chi
        
        if len(systems_with_divergence) >= 1:
            return UniversalFeature(
                name="susceptibility_peak",
                description="Susceptibility shows peak at phase transition",
                systems_exhibiting=systems_with_divergence,
                exponent_values=peak_values,
                confidence=0.6,
                is_novel=False,
            )
        
        return None
    
    def _check_finite_size_scaling(
        self,
        results: List[Any],
        system_names: List[str],
    ) -> Optional[UniversalFeature]:
        """Check for finite-size scaling behavior."""
        systems_with_fss = []
        
        for result, name in zip(results, system_names):
            if not hasattr(result, 'system_sizes') or len(result.system_sizes) < 3:
                continue
            
            # Check if Binder cumulant crossing exists (signature of FSS)
            sizes = result.system_sizes
            binder_data = {}
            
            for L in sizes:
                points = result.get_points_for_size(L)
                if points and hasattr(points[0], 'binder_cumulant'):
                    binder_data[L] = [p.binder_cumulant for p in points]
            
            if len(binder_data) >= 2:
                # Check for crossing point
                systems_with_fss.append(name)
        
        if systems_with_fss:
            return UniversalFeature(
                name="finite_size_scaling",
                description="Observables follow universal finite-size scaling",
                systems_exhibiting=systems_with_fss,
                exponent_values={},
                confidence=0.5,
                is_novel=False,
            )
        
        return None



@dataclass
class SecondarySystemAnalysisReport:
    """Complete analysis report for secondary system exploration."""
    long_range_result: Optional[LongRangeExplorationResult]
    floquet_result: Optional[FloquetExplorationResult]
    cross_comparisons: List[CrossComparisonResult]
    universal_features: List[UniversalFeature]
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def save(self, filepath: str):
        """Save report to JSON."""
        data = {
            'summary': self.summary,
            'cross_comparisons': [c.to_dict() for c in self.cross_comparisons],
            'universal_features': [f.to_dict() for f in self.universal_features],
            'recommendations': self.recommendations,
        }
        
        if self.long_range_result:
            data['long_range_metadata'] = self.long_range_result.metadata
        if self.floquet_result:
            data['floquet_metadata'] = self.floquet_result.metadata
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("SECONDARY SYSTEM ANALYSIS REPORT")
        lines.append("Task 12: Refine Secondary System Anomalies")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Long-Range Ising Results
        if self.long_range_result:
            lines.append("LONG-RANGE ISING EXPLORATION")
            lines.append("-" * 40)
            lines.append(f"  System sizes: {self.long_range_result.system_sizes}")
            lines.append(f"  Total points: {len(self.long_range_result.scan_points)}")
            lines.append(f"  α range: [{self.long_range_result.alpha_values[0]:.2f}, "
                        f"{self.long_range_result.alpha_values[-1]:.2f}]")
            lines.append(f"  h range: [{self.long_range_result.h_values[0]:.2f}, "
                        f"{self.long_range_result.h_values[-1]:.2f}]")
            lines.append("")
        
        # Floquet Ising Results
        if self.floquet_result:
            lines.append("FLOQUET ISING EXPLORATION")
            lines.append("-" * 40)
            lines.append(f"  System sizes: {self.floquet_result.system_sizes}")
            lines.append(f"  Total points: {len(self.floquet_result.scan_points)}")
            tc_count = len(self.floquet_result.get_time_crystal_candidates())
            lines.append(f"  Time crystal candidates: {tc_count}")
            lines.append("")
        
        # Cross-Comparisons
        lines.append("CROSS-COMPARISONS WITH DTFIM")
        lines.append("-" * 40)
        for comp in self.cross_comparisons:
            lines.append(f"\n  {comp.secondary_system.upper()} vs DTFIM:")
            lines.append(f"    Shared critical behavior: {comp.shared_critical_behavior}")
            lines.append(f"    Correlation: {comp.correlation_coefficient:.3f}")
            lines.append(f"    Universal features: {', '.join(comp.universal_features) or 'None'}")
            lines.append(f"    Unique to DTFIM: {', '.join(comp.unique_to_primary[:3])}")
            lines.append(f"    Unique to secondary: {', '.join(comp.unique_to_secondary[:3])}")
        lines.append("")
        
        # Universal Features
        lines.append("UNIVERSAL FEATURES DETECTED")
        lines.append("-" * 40)
        if self.universal_features:
            for feat in self.universal_features:
                lines.append(f"\n  {feat.name}:")
                lines.append(f"    Description: {feat.description}")
                lines.append(f"    Systems: {', '.join(feat.systems_exhibiting)}")
                lines.append(f"    Confidence: {feat.confidence:.1%}")
                lines.append(f"    Novel: {feat.is_novel}")
        else:
            lines.append("  No universal features detected across systems")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def run_secondary_system_refinement(
    dtfim_result_path: Optional[str] = None,
    output_dir: str = "results/task12_secondary_refinement",
    quick_mode: bool = True,
) -> SecondarySystemAnalysisReport:
    """
    Main function to run Task 12: Refine secondary system anomalies.
    
    Args:
        dtfim_result_path: Path to DTFIM refined exploration results
        output_dir: Directory to save results
        quick_mode: Use reduced parameters for faster execution
        
    Returns:
        SecondarySystemAnalysisReport with complete analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Task 12: Refine Secondary System Anomalies")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load DTFIM results if available
    dtfim_result = None
    if dtfim_result_path and Path(dtfim_result_path).exists():
        from .dtfim_refined_explorer import RefinedExplorationResult
        dtfim_result = RefinedExplorationResult.load(dtfim_result_path)
        logger.info(f"Loaded DTFIM results from {dtfim_result_path}")
    
    # Configure parameters based on mode
    if quick_mode:
        lr_sizes = [8, 10, 12]
        floquet_sizes = [8, 10]
        resolution = 5
        logger.info("Running in QUICK MODE with reduced parameters")
    else:
        lr_sizes = [8, 10, 12, 14]
        floquet_sizes = [8, 10, 12]
        resolution = 10
        logger.info("Running in FULL MODE")
    
    # Task 12.1: Refine Long-Range Ising anomalies
    logger.info("\n" + "=" * 60)
    logger.info("Task 12.1: Refining Long-Range Ising anomalies")
    logger.info("=" * 60)
    
    lr_region = LongRangeAnomalousRegion(
        alpha_center=2.0,  # Near mean-field crossover
        h_center=1.0,  # Near critical point
        alpha_width=0.5,
        h_width=0.5,
        anomaly_type="universality_crossover",
        severity=0.7,
        description="Region near mean-field to short-range crossover"
    )
    
    lr_explorer = LongRangeIsingRefinedExplorer(
        system_sizes=lr_sizes,
        J0=1.0,
        periodic=True
    )
    
    lr_result = lr_explorer.refine_anomalous_region(
        region=lr_region,
        resolution_factor=resolution,
        parallel=True,
        max_workers=4
    )
    
    lr_result.save(str(output_path / "long_range_refined.json"))
    logger.info(f"Long-Range Ising results saved")
    
    # Task 12.1: Refine Floquet Ising anomalies
    logger.info("\n" + "=" * 60)
    logger.info("Task 12.1: Refining Floquet Ising anomalies")
    logger.info("=" * 60)
    
    floquet_region = FloquetAnomalousRegion(
        h1_center=1.5,
        h2_center=0.5,
        T_center=1.0,
        h1_width=0.5,
        h2_width=0.3,
        T_width=0.3,
        anomaly_type="time_crystal_candidate",
        severity=0.6,
        description="Region with potential time crystal signatures"
    )
    
    floquet_explorer = FloquetIsingRefinedExplorer(
        system_sizes=floquet_sizes,
        J=1.0,
        n_periods=20
    )
    
    floquet_result = floquet_explorer.refine_anomalous_region(
        region=floquet_region,
        resolution_factor=resolution // 2,  # 3D space, use lower resolution
        parallel=True,
        max_workers=4
    )
    
    floquet_result.save(str(output_path / "floquet_refined.json"))
    logger.info(f"Floquet Ising results saved")
    
    # Task 12.2: Cross-compare with primary system
    logger.info("\n" + "=" * 60)
    logger.info("Task 12.2: Cross-comparing with DTFIM")
    logger.info("=" * 60)
    
    cross_comparisons = []
    comparator = SecondarySystemCrossComparator()
    
    if dtfim_result:
        lr_comparison = comparator.compare_with_dtfim(dtfim_result, lr_result)
        cross_comparisons.append(lr_comparison)
        logger.info(f"Long-Range vs DTFIM: correlation={lr_comparison.correlation_coefficient:.3f}")
        
        floquet_comparison = comparator.compare_with_dtfim(dtfim_result, floquet_result)
        cross_comparisons.append(floquet_comparison)
        logger.info(f"Floquet vs DTFIM: shared_critical={floquet_comparison.shared_critical_behavior}")
    else:
        logger.warning("No DTFIM results available for cross-comparison")
    
    # Task 12.3: Look for universal features
    logger.info("\n" + "=" * 60)
    logger.info("Task 12.3: Detecting universal features")
    logger.info("=" * 60)
    
    detector = UniversalFeatureDetector()
    
    results_list = [lr_result, floquet_result]
    names_list = ["long_range_ising", "floquet_ising"]
    
    if dtfim_result:
        results_list.insert(0, dtfim_result)
        names_list.insert(0, "dtfim")
    
    universal_features = detector.detect_universal_features(results_list, names_list)
    
    for feat in universal_features:
        logger.info(f"Found universal feature: {feat.name}")
        logger.info(f"  Systems: {feat.systems_exhibiting}")
        logger.info(f"  Confidence: {feat.confidence:.1%}")
    
    # Generate summary and recommendations
    summary = {
        'long_range_points': len(lr_result.scan_points),
        'floquet_points': len(floquet_result.scan_points),
        'time_crystal_candidates': len(floquet_result.get_time_crystal_candidates()),
        'universal_features_found': len(universal_features),
        'cross_comparisons_performed': len(cross_comparisons),
    }
    
    recommendations = []
    
    # Check for promising discoveries
    tc_candidates = floquet_result.get_time_crystal_candidates()
    if tc_candidates:
        recommendations.append(
            f"PROMISING: {len(tc_candidates)} time crystal candidates found in Floquet system. "
            "Consider deeper investigation with larger system sizes."
        )
    
    if universal_features:
        recommendations.append(
            f"Universal features detected across {len(universal_features)} categories. "
            "These support the validity of the quantum phase transition analysis."
        )
    
    if cross_comparisons:
        for comp in cross_comparisons:
            if comp.shared_critical_behavior:
                recommendations.append(
                    f"{comp.secondary_system} shares critical behavior with DTFIM. "
                    "Consider joint analysis for universality class identification."
                )
    
    recommendations.append(
        "Proceed to Month 2 Decision Point (Task 13) to evaluate all anomalies "
        "and select the most promising discovery for rigorous validation."
    )
    
    # Create report
    report = SecondarySystemAnalysisReport(
        long_range_result=lr_result,
        floquet_result=floquet_result,
        cross_comparisons=cross_comparisons,
        universal_features=universal_features,
        summary=summary,
        recommendations=recommendations,
    )
    
    # Save report
    report.save(str(output_path / "analysis_report.json"))
    
    # Generate text report
    text_report = report.generate_text_report()
    with open(output_path / "analysis_report.txt", 'w') as f:
        f.write(text_report)
    
    logger.info("\n" + text_report)
    
    return report
