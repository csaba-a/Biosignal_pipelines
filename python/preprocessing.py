import numpy as np
import mne
from scipy import signal
from sklearn.preprocessing import StandardScaler

def bandpass_filter(raw, l_freq, h_freq):
    """Apply bandpass filter to MNE Raw object."""
    return raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)

def notch_filter(raw, freqs):
    """Apply notch filter to remove line noise (e.g., 50/60 Hz)."""
    return raw.copy().notch_filter(freqs=freqs, verbose=False)

def run_ica(raw, n_components=15, random_state=42):
    """Run ICA to remove artifacts (e.g., eye blinks, ECG)."""
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    ica.fit(raw)
    return ica

def apply_ica(raw, ica, exclude=None):
    """Apply ICA to remove selected components."""
    return ica.apply(raw.copy(), exclude=exclude)

def normalize_signals(data):
    """Normalize signals (numpy array or DataFrame) using z-score."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def resample_raw(raw, new_sfreq):
    """Resample MNE Raw object to new sampling frequency."""
    return raw.copy().resample(new_sfreq, npad='auto', verbose=False)

def interpolate_bads(raw, bads):
    """Interpolate bad channels in MNE Raw object."""
    raw.info['bads'] = bads
    return raw.copy().interpolate_bads(reset_bads=True, verbose=False)
