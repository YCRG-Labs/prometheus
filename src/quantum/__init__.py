"""
Quantum simulation infrastructure for exact diagonalization of spin systems.

This module provides tools for:
- Sparse Hamiltonian construction for spin-1/2 chains
- Lanczos algorithm for ground state computation
- Local observables (σx, σy, σz)
- Correlation functions ⟨σᵢσⱼ⟩
- Entanglement entropy (von Neumann)
- Disorder infrastructure for disordered quantum systems
"""

from .hamiltonian_builder import SpinHamiltonianBuilder, SpinChainParams
from .exact_diagonalization import ExactDiagonalizationSolver
from .observables import ObservableCalculator
from .entanglement import EntanglementCalculator
from .disorder import (
    DisorderType,
    DisorderConfig,
    DisorderRealization,
    RandomCouplingGenerator,
    DisorderedSystemConfig,
    DisorderRealizationGenerator,
    DisorderAverageResult,
    DisorderAveragingFramework,
    StatisticalAnalysisResult,
    DisorderStatisticalAnalyzer,
)
from .disordered_tfim import (
    DTFIMParams,
    DisorderedTFIM,
    create_dtfim_uniform_disorder,
    create_dtfim_gaussian_disorder,
)
from .floquet_ising import (
    FloquetIsing,
    FloquetIsingParams,
    FloquetIsingExplorer,
    create_standard_floquet_ising,
)

__all__ = [
    # Hamiltonian building
    'SpinHamiltonianBuilder',
    'SpinChainParams',
    # Exact diagonalization
    'ExactDiagonalizationSolver',
    # Observables
    'ObservableCalculator',
    # Entanglement
    'EntanglementCalculator',
    # Disorder infrastructure
    'DisorderType',
    'DisorderConfig',
    'DisorderRealization',
    'RandomCouplingGenerator',
    'DisorderedSystemConfig',
    'DisorderRealizationGenerator',
    'DisorderAverageResult',
    'DisorderAveragingFramework',
    'StatisticalAnalysisResult',
    'DisorderStatisticalAnalyzer',
    # Disordered TFIM
    'DTFIMParams',
    'DisorderedTFIM',
    'create_dtfim_uniform_disorder',
    'create_dtfim_gaussian_disorder',
    # Floquet Ising
    'FloquetIsing',
    'FloquetIsingParams',
    'FloquetIsingExplorer',
    'create_standard_floquet_ising',
]
