import numpy as np
import mne
from scipy.signal import coherence

def compute_coherence(raw, ch1, ch2, fmin=0.5, fmax=45):
    """Compute coherence between two channels in MNE Raw object."""
    data, times = raw[[ch1, ch2], :]
    f, Cxy = coherence(data[0], data[1], fs=raw.info['sfreq'])
    mask = (f >= fmin) & (f <= fmax)
    return f[mask], Cxy[mask]

def compute_correlation_matrix(raw):
    """Compute correlation matrix between all channels."""
    data = raw.get_data()
    return np.corrcoef(data)

def compute_phase_locking_value(raw, ch1, ch2):
    """Compute phase-locking value (PLV) between two channels."""
    data, _ = raw[[ch1, ch2], :]
    phase1 = np.angle(np.fft.fft(data[0]))
    phase2 = np.angle(np.fft.fft(data[1]))
    plv = np.abs(np.sum(np.exp(1j * (phase1 - phase2)))) / len(phase1)
    return plv
