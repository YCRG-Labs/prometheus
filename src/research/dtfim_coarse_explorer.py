"""
Coarse exploration of Disordered Transverse Field Ising Model (DTFIM).

Performs systematic parameter space scan to identify anomalous regions
and generate preliminary phase diagrams.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from ..quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
from ..quantum.observables import ObservableCalculator
from ..quantum.entanglement import EntanglementCalculator
from ..quantum.disorder import DisorderRealization


@dataclass
class DTFIMScanPoint:
    """Single point in DTFIM parameter space."""
    h: float  # Mean transverse field
    W: float  # Disorder strength
    magnetization_z: float
    magnetization_x: float
    susceptibility_z: float
    susceptibility_x: float
    entanglement_entropy: float
    correlation_length: float
    energy: float
    
    # Statistical measures (from disorder averaging)
    magnetization_z_std: float = 0.0
    susceptibility_z_std: float = 0.0
    entanglement_std: float = 0.0


@dataclass
class AnomalyReport:
    """Report of anomalous behavior in parameter space."""
    h: float
    W: float
    anomaly_type: str
    severity: float  # 0-1 scale
    description: str
    observables: Dict[str, float]


@dataclass
class DTFIMCoarseExplorationResult:
    """Results of coarse DTFIM exploration."""
    scan_points: List[DTFIMScanPoint]
    h_values: np.ndarray
    W_values: np.ndarray
    anomalies: List[AnomalyReport]
    phase_diagram_data: Dict[str, np.ndarray]
    metadata: Dict = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'h_values': self.h_values.tolist(),
            'W_values': self.W_values.tolist(),
            'scan_points': [
                {
                    'h': p.h,
                    'W': p.W,
                    'magnetization_z': p.magnetization_z,
                    'magnetization_x': p.magnetization_x,
                    'susceptibility_z': p.susceptibility_z,
                    'susceptibility_x': p.susceptibility_x,
                    'entanglement_entropy': p.entanglement_entropy,
                    'correlation_length': p.correlation_length,
                    'energy': p.energy,
                    'magnetization_z_std': p.magnetization_z_std,
                    'susceptibility_z_std': p.susceptibility_z_std,
                    'entanglement_std': p.entanglement_std,
                }
                for p in self.scan_points
            ],
            'anomalies': [
                {
                    'h': a.h,
                    'W': a.W,
                    'anomaly_type': a.anomaly_type,
                    'severity': a.severity,
                    'description': a.description,
                    'observables': a.observables,
                }
                for a in self.anomalies
            ],
            'phase_diagram_data': {
                k: v.tolist() for k, v in self.phase_diagram_data.items()
            },
            'metadata': self.metadata,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DTFIMCoarseExplorationResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scan_points = [
            DTFIMScanPoint(**p) for p in data['scan_points']
        ]
        
        anomalies = [
            AnomalyReport(**a) for a in data['anomalies']
        ]
        
        return cls(
            scan_points=scan_points,
            h_values=np.array(data['h_values']),
            W_values=np.array(data['W_values']),
            anomalies=anomalies,
            phase_diagram_data={
                k: np.array(v) for k, v in data['phase_diagram_data'].items()
            },
            metadata=data['metadata']
        )


class DTFIMCoarseExplorer:
    """
    Coarse exploration of DTFIM parameter space.
    
    Systematically scans (h, W) space to:
    - Map out phase diagram
    - Identify anomalous regions
    - Compute key observables
    """
    
    def __init__(
        self,
        L: int = 12,
        n_disorder_realizations: int = 50,
        J_mean: float = 1.0,
        periodic: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize explorer.
        
        Args:
            L: System size
            n_disorder_realizations: Number of disorder realizations per point
            J_mean: Mean coupling strength
            periodic: Use periodic boundary conditions
            random_seed: Random seed for reproducibility
        """
        self.L = L
        self.n_disorder_realizations = n_disorder_realizations
        self.J_mean = J_mean
        self.periodic = periodic
        self.random_seed = random_seed
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_single_point(
        self,
        h: float,
        W: float
    ) -> DTFIMScanPoint:
        """
        Compute observables at a single (h, W) point.
        
        Args:
            h: Mean transverse field
            W: Disorder strength
            
        Returns:
            DTFIMScanPoint with computed observables
        """
        # Create DTFIM
        params = DTFIMParams(
            L=self.L,
            h_mean=h,
            h_disorder=W,
            J_mean=self.J_mean,
            J_disorder=0.0,  # Only disorder in h for now
            periodic=self.periodic
        )
        dtfim = DisorderedTFIM(params)
        
        # Calculators
        obs_calc = ObservableCalculator(self.L)
        ent_calc = EntanglementCalculator(self.L)
        
        # Collect observables over disorder realizations
        energies = []
        mag_z_list = []
        mag_x_list = []
        chi_z_list = []
        chi_x_list = []
        ent_list = []
        corr_len_list = []
        
        # Generate disorder realizations
        for i in range(self.n_disorder_realizations):
            realization = dtfim.disorder_framework.realization_generator.generate_single(
                realization_index=i
            )
            
            # Compute ground state
            E, state = dtfim.compute_ground_state(realization)
            energies.append(E)
            
            # Local observables
            local_obs = obs_calc.local_observables(state)
            mag_z_list.append(abs(local_obs.magnetization_z))
            mag_x_list.append(abs(local_obs.magnetization_x))
            
            # Susceptibilities
            chi_z = obs_calc.susceptibility(state, direction='z')
            chi_x = obs_calc.susceptibility(state, direction='x')
            chi_z_list.append(chi_z)
            chi_x_list.append(chi_x)
            
            # Entanglement (half-chain cut)
            ent = ent_calc.von_neumann_entropy(state, cut_position=self.L // 2)
            ent_list.append(ent)
            
            # Correlation length
            try:
                corr_len = obs_calc.correlation_length(state, correlation_type='zz')
                if np.isfinite(corr_len):
                    corr_len_list.append(corr_len)
            except:
                pass
        
        # Compute statistics
        return DTFIMScanPoint(
            h=h,
            W=W,
            magnetization_z=np.mean(mag_z_list),
            magnetization_x=np.mean(mag_x_list),
            susceptibility_z=np.mean(chi_z_list),
            susceptibility_x=np.mean(chi_x_list),
            entanglement_entropy=np.mean(ent_list),
            correlation_length=np.mean(corr_len_list) if corr_len_list else float('inf'),
            energy=np.mean(energies),
            magnetization_z_std=np.std(mag_z_list),
            susceptibility_z_std=np.std(chi_z_list),
            entanglement_std=np.std(ent_list)
        )
    
    def scan_parameter_space(
        self,
        h_range: Tuple[float, float] = (0.0, 2.0),
        W_range: Tuple[float, float] = (0.0, 2.0),
        n_points: int = 100,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> DTFIMCoarseExplorationResult:
        """
        Scan DTFIM parameter space.
        
        Args:
            h_range: (h_min, h_max) range for transverse field
            W_range: (W_min, W_max) range for disorder strength
            n_points: Total number of points to sample (will be sqrt(n_points) per axis)
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            
        Returns:
            DTFIMCoarseExplorationResult with scan data and anomalies
        """
        # Create grid
        n_per_axis = int(np.sqrt(n_points))
        h_values = np.linspace(h_range[0], h_range[1], n_per_axis)
        W_values = np.linspace(W_range[0], W_range[1], n_per_axis)
        
        self.logger.info(f"Scanning {n_per_axis}x{n_per_axis} grid in (h, W) space")
        self.logger.info(f"h ∈ [{h_range[0]}, {h_range[1]}], W ∈ [{W_range[0]}, {W_range[1]}]")
        self.logger.info(f"System size L={self.L}, {self.n_disorder_realizations} disorder realizations per point")
        
        # Generate all (h, W) pairs
        scan_params = [(h, W) for h in h_values for W in W_values]
        
        # Compute all points
        scan_points = []
        
        if parallel and len(scan_params) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._compute_single_point, h, W): (h, W)
                    for h, W in scan_params
                }
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        point = future.result()
                        scan_points.append(point)
                        if (i + 1) % 10 == 0:
                            self.logger.info(f"Completed {i + 1}/{len(scan_params)} points")
                    except Exception as e:
                        h, W = futures[future]
                        self.logger.error(f"Failed at (h={h}, W={W}): {e}")
        else:
            for i, (h, W) in enumerate(scan_params):
                try:
                    point = self._compute_single_point(h, W)
                    scan_points.append(point)
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Completed {i + 1}/{len(scan_params)} points")
                except Exception as e:
                    self.logger.error(f"Failed at (h={h}, W={W}): {e}")
        
        # Sort by (h, W) for consistency
        scan_points.sort(key=lambda p: (p.h, p.W))
        
        # Generate phase diagram data
        phase_diagram_data = self._generate_phase_diagram_data(
            scan_points, h_values, W_values
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalies(scan_points, phase_diagram_data)
        
        # Create result
        result = DTFIMCoarseExplorationResult(
            scan_points=scan_points,
            h_values=h_values,
            W_values=W_values,
            anomalies=anomalies,
            phase_diagram_data=phase_diagram_data,
            metadata={
                'L': self.L,
                'n_disorder_realizations': self.n_disorder_realizations,
                'J_mean': self.J_mean,
                'periodic': self.periodic,
                'h_range': h_range,
                'W_range': W_range,
                'n_points': len(scan_points),
            }
        )
        
        self.logger.info(f"Scan complete: {len(scan_points)} points, {len(anomalies)} anomalies detected")
        
        return result
    
    def _generate_phase_diagram_data(
        self,
        scan_points: List[DTFIMScanPoint],
        h_values: np.ndarray,
        W_values: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate 2D arrays for phase diagram plotting."""
        n_h = len(h_values)
        n_W = len(W_values)
        
        # Initialize arrays
        mag_z = np.zeros((n_W, n_h))
        mag_x = np.zeros((n_W, n_h))
        chi_z = np.zeros((n_W, n_h))
        chi_x = np.zeros((n_W, n_h))
        ent = np.zeros((n_W, n_h))
        corr_len = np.zeros((n_W, n_h))
        energy = np.zeros((n_W, n_h))
        
        # Fill arrays
        for point in scan_points:
            i_h = np.argmin(np.abs(h_values - point.h))
            i_W = np.argmin(np.abs(W_values - point.W))
            
            mag_z[i_W, i_h] = point.magnetization_z
            mag_x[i_W, i_h] = point.magnetization_x
            chi_z[i_W, i_h] = point.susceptibility_z
            chi_x[i_W, i_h] = point.susceptibility_x
            ent[i_W, i_h] = point.entanglement_entropy
            corr_len[i_W, i_h] = point.correlation_length
            energy[i_W, i_h] = point.energy
        
        return {
            'magnetization_z': mag_z,
            'magnetization_x': mag_x,
            'susceptibility_z': chi_z,
            'susceptibility_x': chi_x,
            'entanglement_entropy': ent,
            'correlation_length': corr_len,
            'energy': energy,
        }
    
    def _detect_anomalies(
        self,
        scan_points: List[DTFIMScanPoint],
        phase_diagram_data: Dict[str, np.ndarray]
    ) -> List[AnomalyReport]:
        """
        Detect anomalous regions in parameter space.
        
        Looks for:
        - Unexpected susceptibility peaks
        - Anomalous entanglement scaling
        - Unusual correlation length behavior
        - Large disorder fluctuations
        """
        anomalies = []
        
        # Extract data
        chi_z = phase_diagram_data['susceptibility_z']
        ent = phase_diagram_data['entanglement_entropy']
        corr_len = phase_diagram_data['correlation_length']
        
        # 1. Susceptibility peaks (potential phase transitions)
        chi_threshold = np.percentile(chi_z[np.isfinite(chi_z)], 90)
        for point in scan_points:
            if point.susceptibility_z > chi_threshold:
                anomalies.append(AnomalyReport(
                    h=point.h,
                    W=point.W,
                    anomaly_type='susceptibility_peak',
                    severity=min(1.0, point.susceptibility_z / chi_threshold - 0.9),
                    description=f'High susceptibility χ={point.susceptibility_z:.3f}',
                    observables={
                        'susceptibility_z': point.susceptibility_z,
                        'magnetization_z': point.magnetization_z,
                        'entanglement': point.entanglement_entropy,
                    }
                ))
        
        # 2. Anomalous entanglement (potential critical regions)
        ent_threshold = np.percentile(ent[np.isfinite(ent)], 85)
        for point in scan_points:
            if point.entanglement_entropy > ent_threshold:
                anomalies.append(AnomalyReport(
                    h=point.h,
                    W=point.W,
                    anomaly_type='high_entanglement',
                    severity=min(1.0, point.entanglement_entropy / ent_threshold - 0.85),
                    description=f'High entanglement S={point.entanglement_entropy:.3f}',
                    observables={
                        'entanglement': point.entanglement_entropy,
                        'susceptibility_z': point.susceptibility_z,
                        'correlation_length': point.correlation_length,
                    }
                ))
        
        # 3. Large disorder fluctuations (Griffiths phase signatures)
        for point in scan_points:
            if point.W > 0.1:  # Only check disordered regions
                # Relative fluctuations
                rel_mag_fluct = point.magnetization_z_std / (point.magnetization_z + 1e-10)
                rel_chi_fluct = point.susceptibility_z_std / (point.susceptibility_z + 1e-10)
                
                if rel_mag_fluct > 0.5 or rel_chi_fluct > 0.5:
                    anomalies.append(AnomalyReport(
                        h=point.h,
                        W=point.W,
                        anomaly_type='large_fluctuations',
                        severity=min(1.0, max(rel_mag_fluct, rel_chi_fluct)),
                        description=f'Large disorder fluctuations (σ_m/m={rel_mag_fluct:.2f})',
                        observables={
                            'magnetization_z': point.magnetization_z,
                            'magnetization_z_std': point.magnetization_z_std,
                            'susceptibility_z_std': point.susceptibility_z_std,
                        }
                    ))
        
        # 4. Diverging correlation length (critical behavior)
        for point in scan_points:
            if np.isfinite(point.correlation_length) and point.correlation_length > self.L / 3:
                anomalies.append(AnomalyReport(
                    h=point.h,
                    W=point.W,
                    anomaly_type='long_correlation_length',
                    severity=min(1.0, point.correlation_length / self.L),
                    description=f'Long correlation length ξ={point.correlation_length:.2f}',
                    observables={
                        'correlation_length': point.correlation_length,
                        'susceptibility_z': point.susceptibility_z,
                        'entanglement': point.entanglement_entropy,
                    }
                ))
        
        # Sort by severity
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        
        return anomalies
