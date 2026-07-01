#!/usr/bin/env python3
"""
Test script for ReadyToFit package.
This script demonstrates basic usage by generating synthetic data
and fitting multiple peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from readytofit import fit_model, plot_fit_result, evaluate_peak_areas, detect_peaks
from readytofit.fit_models import find_best_fit

def test_basic_fit():
    """Test basic multi-peak fitting functionality."""
    print("Testing ReadyToFit package...")

    # Generate synthetic data with two Gaussian peaks
    x = np.linspace(0, 100, 200)
    y_true = (5 * np.exp(-(x - 30)**2 / (2 * 5**2)) +
              3 * np.exp(-(x - 70)**2 / (2 * 3**2)))
    y = y_true + 0.1 * np.random.normal(size=len(x))  # Add noise

    # Define peaks
    peaks = [
        {"model": "gauss"},  # Free parameters
        {"model": "gauss"}
    ]

    # Fit the model
    result = fit_model(x, y, peaks)

    print("Fit successful!")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"Fitted parameters per peak:")
    for i, peak in enumerate(result['params']):
        print(f"  Peak {i+1}: {peak}")
        # Plot results
    fig,ax = plt.subplots(figsize=(10, 6))
    plot_fit_result(x, y, result, show_residual=True, show_rmse=True, fig=fig, ax=ax)
    fig.suptitle("ReadyToFit Test: Two Gaussian Peaks")
    fig.savefig("test_fit.png", dpi=150, bbox_inches='tight')
    print("Plot saved as test_fit.png")


    # Test the usage of all the peak types
    print("\nTesting all peak types...")
    peak_types = ["gauss", "lorentz", "voigt", "asym", "skew", "emg_reversed"]
    for pt in peak_types:
        print(f"  Testing model: {pt}")
        peaks = [{"model": pt}]
        result = fit_model(x, y, peaks)
        print(f"    RMSE: {result['rmse']:.4f}")
    
    # Evaluate areas
    areas = evaluate_peak_areas(x, result)
    print(f"Peak areas: {areas['peaks']}")
    print(f"Total area: {areas['total']:.2f}")
    return result

def test_fixed_parameters():
    """Test fitting with fixed parameters."""
    print("\nTesting fixed parameters...")

    x = np.linspace(0, 100, 200)
    y_true = 4 * np.exp(-(x - 50)**2 / (2 * 4**2))  # Single peak at x=50
    y = y_true + 0.05 * np.random.normal(size=len(x))

    # Fix the peak center at 50
    peaks = [{"model": "gauss", "mu": 50}]

    result = fit_model(x, y, peaks)
    print(f"Fixed mu test - RMSE: {result['rmse']:.4f}")
    print(f"Fitted peak parameters:")
    for i, peak in enumerate(result['params']):
        print(f"  Peak {i+1}: {peak}")

    return result

def test_peak_detection():
    """Test automatic peak detection."""
    print("\nTesting peak detection...")

    # Generate synthetic data with three peaks
    x = np.linspace(0, 100, 300)
    y_true = (
        4 * np.exp(-(x - 20)**2 / (2 * 4**2)) +
        6 * np.exp(-(x - 50)**2 / (2 * 5**2)) +
        3 * np.exp(-(x - 80)**2 / (2 * 3**2))
    )
    y = y_true + 0.15 * np.random.normal(size=len(x))

    # Auto-detect peaks
    detected = detect_peaks(x, y, n_peaks=3)
    print(f"Detected {len(detected)} peaks:")
    for i, peak in enumerate(detected):
        print(f"  Peak {i+1}: μ={peak['mu']:.2f}, A={peak['A']:.3f}, σ={peak['sigma']:.2f}")

    # Define peaks structure (without parameter values - will use detection)
    peaks = [{"model": "gauss"} for _ in detected]

    # Fit the model
    result = fit_model(x, y, peaks, debug=True)
    print(f"Fit successful! RMSE: {result['rmse']:.4f}")
    print(f"Fitted parameter values:")
    for i, peak in enumerate(result['params']):
        print(f"  Peak {i+1}: {peak}")

    return result

def test_find_best_fit():
    """Test the find_best_fit function."""
    print("\nTesting find_best_fit...")

    x = np.linspace(0, 100, 200)
    y_true = (5 * np.exp(-(x - 30)**2 / (2 * 5**2)) +
              3 * np.exp(-(x - 70)**2 / (2 * 3**2)))
    y = y_true + 0.1 * np.random.normal(size=len(x))

    peaks = [{"model": "gauss"}, {"model": "gauss"}]

    best_results = find_best_fit(x, y, peaks, n_iterations=100, n_best_results=2, random_scale=5)
    print(f"Best fit RMSE: {best_results[0][0]:.4f}")
    print(f"Best fit parameters:")
    for i, peak in enumerate(best_results[0][1]['params']):
        print(f"  Peak {i+1}: {peak}")

    return best_results

if __name__ == "__main__":
    try:
        test_basic_fit()
        test_fixed_parameters()
        test_peak_detection()
        test_find_best_fit()
        print("\nAll tests passed! ✅")
    except Exception as e:
        print(f"Test failed: {e}")
        raise