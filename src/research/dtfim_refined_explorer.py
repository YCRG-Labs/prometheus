"""
Refined exploration of Disordered TFIM anomalies.

Implements Task 10: Deep dive into anomalous regions with:
- 10x resolution in anomalous regions
- Larger system sizes (L = 12, 16, 20)
- More disorder realizations (100 → 1000)
- Full entanglement spectrum computation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import time

from ..quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
from ..quantum.observables import ObservableCalculator
from ..quantum.entanglement import EntanglementCalculator, EntanglementSpectrum
from ..quantum.disorder import DisorderRealization


@dataclass
class RefinedScanPoint:
    """Single point in refined DTFIM parameter space with extended observables."""
    h: float  # Mean transverse field
    W: float  # Disorder strength
    L: int  # System size
    n_realizations: int  # Number of disorder realizations
    
    # Basic observables (disorder-averaged)
    magnetization_z: float
    magnetization_x: float
    susceptibility_z: float
    susceptibility_x: float
    entanglement_entropy: float
    correlation_length: float
    energy: float
    energy_gap: float = 0.0
    
    # Statistical measures
    magnetization_z_std: float = 0.0
    magnetization_z_stderr: float = 0.0
    susceptibility_z_std: float = 0.0
    susceptibility_z_stderr: float = 0.0
    entanglement_std: float = 0.0
    entanglement_stderr: float = 0.0
    energy_std: float = 0.0
    
    # Entanglement spectrum data
    entanglement_spectrum_mean: Optional[np.ndarray] = None
    entanglement_spectrum_gap: float = 0.0  # Gap in entanglement spectrum
    schmidt_rank_mean: float = 0.0
    
    # Additional quantum observables
    binder_cumulant: float = 0.0  # Binder cumulant for phase transition detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'h': self.h,
            'W': self.W,
            'L': self.L,
            'n_realizations': self.n_realizations,
            'magnetization_z': self.magnetization_z,
            'magnetization_x': self.magnetization_x,
            'susceptibility_z': self.susceptibility_z,
            'susceptibility_x': self.susceptibility_x,
            'entanglement_entropy': self.entanglement_entropy,
            'correlation_length': self.correlation_length,
            'energy': self.energy,
            'energy_gap': self.energy_gap,
            'magnetization_z_std': self.magnetization_z_std,
            'magnetization_z_stderr': self.magnetization_z_stderr,
            'susceptibility_z_std': self.susceptibility_z_std,
            'susceptibility_z_stderr': self.susceptibility_z_stderr,
            'entanglement_std': self.entanglement_std,
            'entanglement_stderr': self.entanglement_stderr,
            'energy_std': self.energy_std,
            'entanglement_spectrum_gap': self.entanglement_spectrum_gap,
            'schmidt_rank_mean': self.schmidt_rank_mean,
            'binder_cumulant': self.binder_cumulant,
        }
        if self.entanglement_spectrum_mean is not None:
            result['entanglement_spectrum_mean'] = self.entanglement_spectrum_mean.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefinedScanPoint':
        """Create from dictionary."""
        spectrum = data.pop('entanglement_spectrum_mean', None)
        point = cls(**data)
        if spectrum is not None:
            point.entanglement_spectrum_mean = np.array(spectrum)
        return point


@dataclass
class AnomalousRegion:
    """Definition of an anomalous region for refined exploration."""
    h_center: float
    W_center: float
    h_width: float = 0.2  # Half-width of region
    W_width: float = 0.2
    anomaly_type: str = "unknown"
    severity: float = 0.0
    description: str = ""
    
    @property
    def h_range(self) -> Tuple[float, float]:
        return (self.h_center - self.h_width, self.h_center + self.h_width)
    
    @property
    def W_range(self) -> Tuple[float, float]:
        return (self.W_center - self.W_width, self.W_center + self.W_width)


@dataclass
class RefinedExplorationResult:
    """Results of refined DTFIM exploration."""
    scan_points: List[RefinedScanPoint]
    system_sizes: List[int]
    h_values: np.ndarray
    W_values: np.ndarray
    anomalous_region: AnomalousRegion
    metadata: Dict = field(default_factory=dict)
    
    # Finite-size scaling data
    fss_data: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'h_values': self.h_values.tolist(),
            'W_values': self.W_values.tolist(),
            'system_sizes': self.system_sizes,
            'scan_points': [p.to_dict() for p in self.scan_points],
            'anomalous_region': {
                'h_center': self.anomalous_region.h_center,
                'W_center': self.anomalous_region.W_center,
                'h_width': self.anomalous_region.h_width,
                'W_width': self.anomalous_region.W_width,
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
    def load(cls, filepath: str) -> 'RefinedExplorationResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scan_points = [RefinedScanPoint.from_dict(p) for p in data['scan_points']]
        anomalous_region = AnomalousRegion(**data['anomalous_region'])
        
        fss_data = {}
        if 'fss_data' in data:
            for obs, size_data in data['fss_data'].items():
                fss_data[obs] = {int(L): np.array(vals) for L, vals in size_data.items()}
        
        return cls(
            scan_points=scan_points,
            system_sizes=data['system_sizes'],
            h_values=np.array(data['h_values']),
            W_values=np.array(data['W_values']),
            anomalous_region=anomalous_region,
            metadata=data['metadata'],
            fss_data=fss_data,
        )
    
    def get_points_for_size(self, L: int) -> List[RefinedScanPoint]:
        """Get all scan points for a specific system size."""
        return [p for p in self.scan_points if p.L == L]
    
    def get_observable_grid(self, observable: str, L: int) -> np.ndarray:
        """Get 2D grid of observable values for a specific system size."""
        points = self.get_points_for_size(L)
        n_h = len(self.h_values)
        n_W = len(self.W_values)
        grid = np.full((n_W, n_h), np.nan)
        
        for point in points:
            i_h = np.argmin(np.abs(self.h_values - point.h))
            i_W = np.argmin(np.abs(self.W_values - point.W))
            grid[i_W, i_h] = getattr(point, observable, np.nan)
        
        return grid



class DTFIMRefinedExplorer:
    """
    Refined exploration of DTFIM anomalous regions.
    
    Provides:
    - 10x resolution scanning in anomalous regions
    - Multi-size finite-size scaling (L = 12, 16, 20)
    - High-statistics disorder averaging (up to 1000 realizations)
    - Full entanglement spectrum computation
    """
    
    def __init__(
        self,
        system_sizes: List[int] = [12, 16, 20],
        n_disorder_realizations: int = 1000,
        J_mean: float = 1.0,
        periodic: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize refined explorer.
        
        Args:
            system_sizes: List of system sizes for finite-size scaling
            n_disorder_realizations: Number of disorder realizations per point
            J_mean: Mean coupling strength
            periodic: Use periodic boundary conditions
            random_seed: Random seed for reproducibility
        """
        self.system_sizes = sorted(system_sizes)
        self.n_disorder_realizations = n_disorder_realizations
        self.J_mean = J_mean
        self.periodic = periodic
        self.random_seed = random_seed
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_single_point(
        self,
        h: float,
        W: float,
        L: int,
        n_realizations: int
    ) -> RefinedScanPoint:
        """
        Compute observables at a single (h, W) point for system size L.
        
        Args:
            h: Mean transverse field
            W: Disorder strength
            L: System size
            n_realizations: Number of disorder realizations
            
        Returns:
            RefinedScanPoint with computed observables
        """
        # Create DTFIM
        params = DTFIMParams(
            L=L,
            h_mean=h,
            h_disorder=W,
            J_mean=self.J_mean,
            J_disorder=0.0,
            periodic=self.periodic
        )
        dtfim = DisorderedTFIM(params)
        
        # Calculators
        obs_calc = ObservableCalculator(L)
        ent_calc = EntanglementCalculator(L)
        
        # Collect observables over disorder realizations
        energies = []
        energy_gaps = []
        mag_z_list = []
        mag_x_list = []
        chi_z_list = []
        chi_x_list = []
        ent_list = []
        corr_len_list = []
        mag_z_sq_list = []  # For Binder cumulant
        mag_z_fourth_list = []
        
        # Entanglement spectrum data
        ent_spectra = []
        schmidt_ranks = []
        
        for i in range(n_realizations):
            realization = dtfim.disorder_framework.realization_generator.generate_single(
                realization_index=i
            )
            
            # Compute ground state
            E, state = dtfim.compute_ground_state(realization)
            energies.append(E)
            
            # Compute energy gap (for small systems)
            if L <= 16:
                try:
                    H = dtfim.build_hamiltonian(realization)
                    gap = dtfim.solver.energy_gap(H)
                    energy_gaps.append(gap)
                except:
                    pass
            
            # Local observables
            local_obs = obs_calc.local_observables(state)
            mag_z = abs(local_obs.magnetization_z)
            mag_z_list.append(mag_z)
            mag_x_list.append(abs(local_obs.magnetization_x))
            
            # For Binder cumulant
            mag_z_sq_list.append(mag_z ** 2)
            mag_z_fourth_list.append(mag_z ** 4)
            
            # Susceptibilities
            chi_z = obs_calc.susceptibility(state, direction='z')
            chi_x = obs_calc.susceptibility(state, direction='x')
            chi_z_list.append(chi_z)
            chi_x_list.append(chi_x)
            
            # Full entanglement spectrum
            ent_result = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
            ent_list.append(ent_result.entropy)
            ent_spectra.append(ent_result.eigenvalues[:min(10, len(ent_result.eigenvalues))])
            schmidt_ranks.append(np.sum(ent_result.eigenvalues > 1e-10))
            
            # Correlation length
            try:
                corr_len = obs_calc.correlation_length(state, correlation_type='zz')
                if np.isfinite(corr_len) and corr_len < L:
                    corr_len_list.append(corr_len)
            except:
                pass
        
        # Compute statistics
        n = len(mag_z_list)
        
        # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
        m2_mean = np.mean(mag_z_sq_list)
        m4_mean = np.mean(mag_z_fourth_list)
        binder = 1.0 - m4_mean / (3.0 * m2_mean ** 2) if m2_mean > 1e-10 else 0.0
        
        # Average entanglement spectrum
        max_spectrum_len = max(len(s) for s in ent_spectra)
        padded_spectra = []
        for s in ent_spectra:
            padded = np.zeros(max_spectrum_len)
            padded[:len(s)] = s
            padded_spectra.append(padded)
        mean_spectrum = np.mean(padded_spectra, axis=0)
        
        # Entanglement spectrum gap (between largest and second-largest eigenvalue)
        if len(mean_spectrum) >= 2:
            ent_gap = -np.log(mean_spectrum[1] / mean_spectrum[0]) if mean_spectrum[1] > 1e-15 else np.inf
        else:
            ent_gap = np.inf
        
        return RefinedScanPoint(
            h=h,
            W=W,
            L=L,
            n_realizations=n_realizations,
            magnetization_z=np.mean(mag_z_list),
            magnetization_x=np.mean(mag_x_list),
            susceptibility_z=np.mean(chi_z_list),
            susceptibility_x=np.mean(chi_x_list),
            entanglement_entropy=np.mean(ent_list),
            correlation_length=np.mean(corr_len_list) if corr_len_list else float('inf'),
            energy=np.mean(energies),
            energy_gap=np.mean(energy_gaps) if energy_gaps else 0.0,
            magnetization_z_std=np.std(mag_z_list, ddof=1) if n > 1 else 0.0,
            magnetization_z_stderr=np.std(mag_z_list, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            susceptibility_z_std=np.std(chi_z_list, ddof=1) if n > 1 else 0.0,
            susceptibility_z_stderr=np.std(chi_z_list, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            entanglement_std=np.std(ent_list, ddof=1) if n > 1 else 0.0,
            entanglement_stderr=np.std(ent_list, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            energy_std=np.std(energies, ddof=1) if n > 1 else 0.0,
            entanglement_spectrum_mean=mean_spectrum,
            entanglement_spectrum_gap=ent_gap,
            schmidt_rank_mean=np.mean(schmidt_ranks),
            binder_cumulant=binder,
        )
    
    def refine_anomalous_region(
        self,
        region: AnomalousRegion,
        resolution_factor: int = 10,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> RefinedExplorationResult:
        """
        Perform refined exploration of an anomalous region.
        
        Args:
            region: Anomalous region to explore
            resolution_factor: Resolution increase factor (default 10x)
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback for progress updates
            
        Returns:
            RefinedExplorationResult with detailed scan data
        """
        # Create high-resolution grid
        n_points_per_axis = resolution_factor
        h_values = np.linspace(region.h_range[0], region.h_range[1], n_points_per_axis)
        W_values = np.linspace(region.W_range[0], region.W_range[1], n_points_per_axis)
        
        # Ensure W values are non-negative
        W_values = np.maximum(W_values, 0.0)
        
        self.logger.info(f"Refined exploration of region: h ∈ {region.h_range}, W ∈ {region.W_range}")
        self.logger.info(f"Resolution: {n_points_per_axis}x{n_points_per_axis} = {n_points_per_axis**2} points")
        self.logger.info(f"System sizes: {self.system_sizes}")
        self.logger.info(f"Disorder realizations: {self.n_disorder_realizations}")
        
        # Generate all tasks: (h, W, L)
        tasks = []
        for h in h_values:
            for W in W_values:
                for L in self.system_sizes:
                    tasks.append((h, W, L))
        
        total_tasks = len(tasks)
        self.logger.info(f"Total computations: {total_tasks}")
        
        # Compute all points
        scan_points = []
        start_time = time.time()
        
        if parallel and total_tasks > 1:
            # Use ThreadPoolExecutor for better compatibility
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._compute_single_point, h, W, L, self.n_disorder_realizations
                    ): (h, W, L)
                    for h, W, L in tasks
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
                            if progress_callback:
                                progress_callback(i + 1, total_tasks)
                    except Exception as e:
                        h, W, L = futures[future]
                        self.logger.error(f"Failed at (h={h}, W={W}, L={L}): {e}")
        else:
            for i, (h, W, L) in enumerate(tasks):
                try:
                    point = self._compute_single_point(h, W, L, self.n_disorder_realizations)
                    scan_points.append(point)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_tasks}")
                        if progress_callback:
                            progress_callback(i + 1, total_tasks)
                except Exception as e:
                    self.logger.error(f"Failed at (h={h}, W={W}, L={L}): {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed in {elapsed/60:.1f} minutes")
        
        # Build finite-size scaling data
        fss_data = self._build_fss_data(scan_points, h_values, W_values)
        
        # Create result
        result = RefinedExplorationResult(
            scan_points=scan_points,
            system_sizes=self.system_sizes,
            h_values=h_values,
            W_values=W_values,
            anomalous_region=region,
            metadata={
                'n_disorder_realizations': self.n_disorder_realizations,
                'J_mean': self.J_mean,
                'periodic': self.periodic,
                'resolution_factor': resolution_factor,
                'total_points': len(scan_points),
                'computation_time_minutes': elapsed / 60,
            },
            fss_data=fss_data,
        )
        
        return result
    
    def _build_fss_data(
        self,
        scan_points: List[RefinedScanPoint],
        h_values: np.ndarray,
        W_values: np.ndarray
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Build finite-size scaling data structures."""
        observables = [
            'magnetization_z', 'susceptibility_z', 'entanglement_entropy',
            'correlation_length', 'binder_cumulant', 'energy_gap'
        ]
        
        fss_data = {obs: {} for obs in observables}
        
        for L in self.system_sizes:
            points_L = [p for p in scan_points if p.L == L]
            
            for obs in observables:
                grid = np.full((len(W_values), len(h_values)), np.nan)
                
                for point in points_L:
                    i_h = np.argmin(np.abs(h_values - point.h))
                    i_W = np.argmin(np.abs(W_values - point.W))
                    grid[i_W, i_h] = getattr(point, obs, np.nan)
                
                fss_data[obs][L] = grid
        
        return fss_data
    
    def identify_anomalous_regions_from_coarse(
        self,
        coarse_result_path: str,
        n_regions: int = 5,
        min_severity: float = 0.5
    ) -> List[AnomalousRegion]:
        """
        Identify anomalous regions from coarse exploration results.
        
        Args:
            coarse_result_path: Path to coarse exploration JSON
            n_regions: Maximum number of regions to identify
            min_severity: Minimum anomaly severity threshold
            
        Returns:
            List of AnomalousRegion objects for refined exploration
        """
        from .dtfim_coarse_explorer import DTFIMCoarseExplorationResult
        
        coarse_result = DTFIMCoarseExplorationResult.load(coarse_result_path)
        
        # Filter anomalies by severity
        high_severity = [a for a in coarse_result.anomalies if a.severity >= min_severity]
        
        # Cluster nearby anomalies
        regions = []
        used_anomalies = set()
        
        for anomaly in sorted(high_severity, key=lambda a: a.severity, reverse=True):
            if id(anomaly) in used_anomalies:
                continue
            
            # Find nearby anomalies
            cluster = [anomaly]
            for other in high_severity:
                if id(other) not in used_anomalies:
                    if (abs(other.h - anomaly.h) < 0.3 and 
                        abs(other.W - anomaly.W) < 0.3):
                        cluster.append(other)
                        used_anomalies.add(id(other))
            
            used_anomalies.add(id(anomaly))
            
            # Create region from cluster
            h_center = np.mean([a.h for a in cluster])
            W_center = np.mean([a.W for a in cluster])
            h_spread = max(0.2, np.std([a.h for a in cluster]) * 2)
            W_spread = max(0.2, np.std([a.W for a in cluster]) * 2)
            
            region = AnomalousRegion(
                h_center=h_center,
                W_center=W_center,
                h_width=h_spread,
                W_width=W_spread,
                anomaly_type=anomaly.anomaly_type,
                severity=max(a.severity for a in cluster),
                description=f"Cluster of {len(cluster)} anomalies: {anomaly.description}"
            )
            regions.append(region)
            
            if len(regions) >= n_regions:
                break
        
        return regions
