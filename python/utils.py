import pandas as pd
import mne
import os

def load_csv(file_path):
    """Load biosignal data from a CSV file."""
    return pd.read_csv(file_path)

def load_edf(file_path):
    """Load biosignal data from an EDF file using MNE."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw

def get_sampling_rate(raw):
    """Extract sampling rate from MNE Raw object."""
    return raw.info['sfreq']

def get_channel_names(raw):
    """Extract channel names from MNE Raw object."""
    return raw.info['ch_names']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
