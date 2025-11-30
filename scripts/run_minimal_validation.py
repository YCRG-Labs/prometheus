#!/usr/bin/env python3
"""
Minimal validation script for DTFIM Griffiths phase.
Uses very small parameters to run quickly while demonstrating real physics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_minimal_validation():
    """Run minimal but real validation."""
    
    print("=" * 70)
    print("MINIMAL REAL VALIDATION - DTFIM GRIFFITHS PHASE")
    print("=" * 70)
    print()
    
    # Import quantum modules
    from src.quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
    from src.quantum.observables import ObservableCalculator
    from src.quantum.entanglement import EntanglementCalculator
    
    # Parameters - very small for speed
    system_sizes = [6, 8, 10]
    h_values = np.linspace(0.5, 1.5, 7)
    W = 0.5  # Disorder strength
    n_realizations = 10  # Small for speed
    
    print(f"System sizes: {system_sizes}")
    print(f"h values: {h_values}")
    print(f"Disorder W: {W}")
    print(f"Realizations: {n_realizations}")
    print()
    
    # Collect data
    results = {}
    
    for L in system_sizes:
        print(f"\n--- L = {L} ---")
        results[L] = {'h': [], 'mag': [], 'chi': [], 'entropy': [], 'gap': []}
        
        params = DTFIMParams(
            L=L, h_mean=1.0, h_disorder=W,
            J_mean=1.0, J_disorder=0.0, periodic=True
        )
        
        obs_calc = ObservableCalculator(L)
        ent_calc = EntanglementCalculator(L)
        
        for h in h_values:
            params.h_mean = h
            dtfim = DisorderedTFIM(params)
            
            mag_list, chi_list, ent_list, gap_list = [], [], [], []
            
            for i in range(n_realizations):
                realization = dtfim.disorder_framework.realization_generator.generate_single(i)
                E, state = dtfim.compute_ground_state(realization)
                
                # Magnetization
                local_obs = obs_calc.local_observables(state)
                mag_list.append(abs(local_obs.magnetization_z))
                
                # Susceptibility
                chi = obs_calc.susceptibility(state, direction='z')
                chi_list.append(chi)
                
                # Entanglement
                ent = ent_calc.half_chain_entropy(state)
                ent_list.append(ent)
                
                # Energy gap
                try:
                    H = dtfim.build_hamiltonian(realization)
                    gap = dtfim.solver.energy_gap(H)
                    if gap > 0:
                        gap_list.append(gap)
                except:
                    pass
            
            results[L]['h'].append(h)
            results[L]['mag'].append(np.mean(mag_list))
            results[L]['chi'].append(np.mean(chi_list))
            results[L]['entropy'].append(np.mean(ent_list))
            results[L]['gap'].append(np.mean(gap_list) if gap_list else 0)
            
            print(f"  h={h:.2f}: m={np.mean(mag_list):.3f}, χ={np.mean(chi_list):.2f}, S={np.mean(ent_list):.3f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    # Find critical point from susceptibility peak
    print("\n1. CRITICAL POINT DETECTION")
    for L in system_sizes:
        chi_arr = np.array(results[L]['chi'])
        h_arr = np.array(results[L]['h'])
        max_idx = np.argmax(chi_arr)
        hc = h_arr[max_idx]
        print(f"   L={L}: hc ≈ {hc:.3f} (χ_max = {chi_arr[max_idx]:.2f})")
    
    # Dynamical exponent from gap scaling
    print("\n2. DYNAMICAL EXPONENT z")
    L_arr = np.array(system_sizes)
    gap_at_hc = []
    for L in system_sizes:
        chi_arr = np.array(results[L]['chi'])
        gap_arr = np.array(results[L]['gap'])
        max_idx = np.argmax(chi_arr)
        gap_at_hc.append(gap_arr[max_idx])
    
    gap_arr = np.array(gap_at_hc)
    if np.all(gap_arr > 0):
        log_L = np.log(L_arr)
        log_gap = np.log(gap_arr)
        from scipy.stats import linregress
        slope, intercept, r, p, se = linregress(log_L, log_gap)
        z = -slope
        print(f"   z = {z:.2f} ± {se:.2f} (R² = {r**2:.3f})")
        print(f"   Gap scaling: Δ ~ L^(-{z:.2f})")
    
    # Entanglement scaling
    print("\n3. ENTANGLEMENT ENTROPY SCALING")
    S_at_hc = []
    for L in system_sizes:
        chi_arr = np.array(results[L]['chi'])
        ent_arr = np.array(results[L]['entropy'])
        max_idx = np.argmax(chi_arr)
        S_at_hc.append(ent_arr[max_idx])
    
    S_arr = np.array(S_at_hc)
    log_L = np.log(L_arr)
    slope, intercept, r, p, se = linregress(log_L, S_arr)
    c = 3 * slope  # Central charge from S = (c/3) log(L)
    print(f"   S(L) = {slope:.3f} * log(L) + {intercept:.3f}")
    print(f"   Central charge c ≈ {c:.2f} (R² = {r**2:.3f})")
    
    # Validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Check if results are consistent with Griffiths physics
    is_griffiths = z > 2.0 if 'z' in dir() else False
    is_critical = c > 0.3 if 'c' in dir() else False
    
    print(f"\n   Dynamical exponent z: {z:.2f}" if 'z' in dir() else "   z: not computed")
    print(f"   Central charge c: {c:.2f}" if 'c' in dir() else "   c: not computed")
    print()
    
    if is_griffiths:
        print("   ✓ Large z suggests Griffiths-like physics")
    else:
        print("   ✗ z not anomalously large")
    
    if is_critical:
        print("   ✓ Logarithmic entanglement scaling (critical)")
    else:
        print("   ✗ Entanglement scaling unclear")
    
    print("\n" + "=" * 70)
    print("NOTE: This is a minimal validation with small parameters.")
    print("Full validation requires larger systems and more realizations.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_minimal_validation()
