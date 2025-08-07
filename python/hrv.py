import neurokit2 as nk
import pandas as pd

def compute_hrv(ecg_signal, sampling_rate):
    """Compute HRV features from an ECG signal using neurokit2."""
    processed = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    hrv_indices = nk.hrv(processed[1], sampling_rate=sampling_rate)
    return hrv_indices
