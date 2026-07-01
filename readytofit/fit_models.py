from scipy.optimize import curve_fit
import numpy as np
from .models import build_model
from .parameters import flatten_params, flatten_bounds, unflatten_params, generate_default_p0, validate_bounds, generate_default_bounds
from .peak_detection import estimate_initial_parameters
from typing import List, Dict, Tuple, Callable, Optional

def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict],
    p0: Optional[List[Dict]] = None,
    bounds: Optional[List[Dict]] = None,
    debug: bool = False
) -> Dict:
    """
    Fit a multi-peak model to data using scipy.optimize.curve_fit.

    This function builds a composite model from multiple peak definitions
    and fits it to the provided (x, y) data.

    Parameters
    ----------
    x : array-like
        Independent variable data.

    y : array-like
        Dependent variable data.

    peaks : list of dict
        List of peak definitions. Each dict must include:
            - "model": str
                Model type ("gauss", "voigt", "asym", "skew")

        Optional:
            - "mu": float
                If provided, fixes the peak center (μ is not fitted).

    p0 : list of dict, optional
        Initial parameter guess (one dict per peak).
        Each dict maps param_name -> initial_value.
        Can contain None values or missing keys - auto-filled.

    bounds : list of dict, optional
        Bounds for parameters (one dict per peak).
        Each dict maps param_name -> (lower, upper).
        Use None entries for defaults.
        To fix a parameter: set lower == upper (but scipy requires lower < upper,
        so use v-1e-12 and v+1e-12 instead).

    debug : bool, default=False
        If True, prints fitted parameter values.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "popt": optimized parameters (flat list, free parameters only)
        - "params": structured parameters per peak (list of dicts with all parameters including fixed ones)
        - "param_names": names of free parameters
        - "total_fit": full fitted curve
        - "peak_fits": list of individual peak curves
        - "residual": y - total_fit
        - "rmse": root mean square error
        - "p0": final initial guess used (structured)
        - "bounds": final bounds used (structured)
        - "model_function": callable model
        - "param_slices": parameter index ranges per peak
    """

    # ---- Build composite model ----
    model_fun, param_slices, param_names = build_model(peaks)

    # ---- Initial guess handling ----
    if p0 is None:
        # Use intelligent peak detection to estimate initial parameters
        if debug:
            print("Using peak detection for initial parameter estimation...")
        peaks_estimated = estimate_initial_parameters(x, y, peaks)
        p0 = peaks_estimated
    
    final_p0 = flatten_params(peaks, p0)

    # Fallback if invalid p0
    if final_p0 is None:
        if debug:
            print("Invalid p0 - falling back to manual default guess")
        peaks_estimated = estimate_initial_parameters(x, y, peaks)
        p0 = peaks_estimated
        final_p0 = flatten_params(peaks, p0)

    # ---- Bounds handling ----
    validated_bounds = validate_bounds(bounds, peaks)

    if validated_bounds is None:
        final_bounds_structured = generate_default_bounds(peaks)
    else:
        final_bounds_structured = validated_bounds

    # Convert structured bounds to flat format for curve_fit
    final_bounds = flatten_bounds(peaks, final_bounds_structured)

    # ---- Perform fit ----
    popt, _ = curve_fit(
        model_fun,
        x,
        y,
        p0=final_p0,
        bounds=final_bounds,
        maxfev=10000  # Increase max function evaluations for complex fits
    )

    # ---- Compute fitted curves ----
    total_fit = model_fun(x, *popt)

    # Individual peak contributions
    peak_fits = []
    for (start, end), peak in zip(param_slices, peaks):
        sub_params = popt[start:end]

        # Build single-peak model
        single_model, _, _ = build_model([peak])
        peak_fits.append(single_model(x, *sub_params))

    # Residuals
    residual = y - total_fit

    # ---- Unflatten parameters back to structured format ----
    params_structured = unflatten_params(peaks, popt)

    # ---- Package results ----
    result = {
        "popt": popt,
        "params": params_structured,  # Structured parameters with fixed values included
        "param_names": param_names,
        "total_fit": total_fit,
        "peak_fits": peak_fits,
        "residual": residual,
        "rmse": np.sqrt(np.mean(residual**2)),
        "p0": final_p0,
        "bounds": final_bounds_structured,
        "model_function": model_fun,
        "param_slices": param_slices,
    }

    # ---- Debug output ----
    if debug:
        print("\nFitted parameters:")
        for name, value in zip(param_names, popt):
            print(f"{name}: {value}")

    return result


def find_best_fit(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict],
    p0: Optional[List[Dict]] = None,
    bounds: Optional[List[Dict]] = None,
    n_iterations: int = 10,
    n_best_results: int = 3,
    random_scale: float = 0.1,
    debug: bool = False
) -> list:
    """
    Fit the model and return the best fit result.
    Currently, this function simply calls fit_model, but it can be extended
    to perform multiple fits with different initial guesses or models and select the best one.
    """
    best_res = [100]    
    if p0 is None:
        if debug:
            print("No initial guess provided. Using peak detection for initial parameter estimation...")
        peaks_estimated = estimate_initial_parameters(x, y, peaks)
        p0 = peaks_estimated
    else:
        if debug:
            print("Using provided initial guess for fitting.")
    p0_flat = flatten_params(peaks, p0)
    
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}...")
        if i>0:
           p0_flat = perturb_p0(p0_flat, random_scale, bounds, peaks)  # Perturb initial guess for next iteration
        
        p0 = unflatten_params(peaks, p0_flat)  # Ensure p0 is structured for the first iteration
        result = fit_model(x, y, peaks, p0, bounds, debug)
        if result['rmse'] < best_res[-1]:   #if current RMSE is better than worst in best_res
            print(f"\tCurrent RMSE: {result['rmse']:.4f} - Adding to best results")
            if len(best_res) == n_best_results:
                best_res.pop()
            best_res.append(result['rmse'])
            best_res.sort(reverse=True)   # Sort by RMSE
        else:
            print(f"\tCurrent RMSE: {result['rmse']:.4f} - Not better than worst of best RMSE: {best_res[-1]:.4f}")
            continue
    return best_res

def perturb_p0(p0: list, scale: float = 0.1, bounds: Optional[List[Dict]] = None, 
               peaks: Optional[List[Dict]] = None, debug: bool = False) -> list:
    """
    Perturb the initial guess parameters by a random factor within a specified scale.
    The perturbation respects the parameter bounds to ensure the perturbed values
    remain within valid ranges.
    
    This can help escape local minima in optimization while maintaining valid parameter values.
    
    Parameters
    ----------
    p0 : list
        Flat list of initial parameter values.
    scale : float
        Scale of the perturbation (as a fraction of the parameter value).
    bounds : list of dict, optional
        Structured bounds (one dict per peak, mapping param_name -> (lower, upper)).
        If provided, perturbed values will be clamped to stay within these bounds.
    peaks : list of dict, optional
        Peak definitions (required if bounds is provided, to determine parameter order).
    debug : bool
        If True, print debug information.
        
    Returns
    -------
    perturbed_p0 : list
        Perturbed parameter values, clamped to bounds if bounds were provided.
    """
    # Flatten bounds to match the flat p0 format
    lower_bounds = None
    upper_bounds = None
    
    if bounds is not None and peaks is not None:
        flattened = flatten_bounds(peaks, bounds)
        if flattened is not None:
            lower_bounds = np.array(flattened[0])
            upper_bounds = np.array(flattened[1])
    
    perturbed_p0 = []
    for i, param in enumerate(p0):
        if debug:
            print(f"Original p0[{i}]: {param}")
        if param is None:
            perturbed_p0.append(None)
        else:
            # Calculate perturbation
            perturbation = np.random.uniform(-scale, scale) * abs(param)
            new_val = param + perturbation
            
            # Clamp to bounds if available
            if lower_bounds is not None and upper_bounds is not None:
                if i < len(lower_bounds):
                    lb = lower_bounds[i]
                    ub = upper_bounds[i]
                    # Only clamp if bounds are finite
                    if np.isfinite(lb) and np.isfinite(ub):
                        new_val = max(lb, min(ub, new_val))
                    elif np.isfinite(lb):
                        new_val = max(lb, new_val)
                    elif np.isfinite(ub):
                        new_val = min(ub, new_val)
            
            perturbed_p0.append(new_val)
            if debug:
                print(f"Perturbed p0[{i}]: {perturbed_p0[-1]} (perturbation: {perturbation:.4f})")
    
    return perturbed_p0
