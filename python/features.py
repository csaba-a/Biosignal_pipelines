import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from neurokit2 import hrv, ecg_process

def extract_time_features(signal):
    """Extract time-domain features from a 1D numpy array."""
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'var': np.var(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'zero_crossings': ((signal[:-1] * signal[1:]) < 0).sum(),
    }

def extract_freq_features(signal, sfreq):
    """Extract frequency-domain features using Welch's method."""
    freqs, psd = welch(signal, sfreq, nperseg=2*sfreq)
    total_power = np.sum(psd)
    band_powers = {
        'delta': bandpower(psd, freqs, 0.5, 4),
        'theta': bandpower(psd, freqs, 4, 8),
        'alpha': bandpower(psd, freqs, 8, 13),
        'beta': bandpower(psd, freqs, 13, 30),
        'gamma': bandpower(psd, freqs, 30, 45),
    }
    return {'total_power': total_power, **band_powers}

def bandpower(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.sum(psd[mask])

def extract_nonlinear_features(signal):
    """Extract nonlinear features (entropy, fractal dimension, etc.)."""
    try:
        import antropy
        return {
            'sample_entropy': antropy.sample_entropy(signal),
            'spectral_entropy': antropy.spectral_entropy(signal, sf=1000, method='welch'),
            'dfa': antropy.detrended_fluctuation(signal),
        }
    except ImportError:
        return {}

def extract_features_df(df, sfreq):
    """Extract features for each column in a DataFrame."""
    features = {}
    for col in df.columns:
        sig = df[col].values
        features[col+'_time'] = extract_time_features(sig)
        features[col+'_freq'] = extract_freq_features(sig, sfreq)
        features[col+'_nonlinear'] = extract_nonlinear_features(sig)
    return features
