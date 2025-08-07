import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne

def plot_raw_signal(raw, channels=None, start=0, duration=10):
    """Plot raw signals for selected channels and time window."""
    if channels is None:
        channels = raw.ch_names[:5]
    raw.plot(start=start, duration=duration, picks=channels, show=True)

def plot_psd(raw, fmin=0.5, fmax=45):
    """Plot power spectral density for all channels."""
    raw.plot_psd(fmin=fmin, fmax=fmax, show=True)

def plot_hrv(hrv_df):
    """Plot HRV time series from a DataFrame."""
    plt.figure(figsize=(10, 4))
    plt.plot(hrv_df['HRV_MeanNN'])
    plt.title('HRV MeanNN Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MeanNN (ms)')
    plt.tight_layout()
    plt.show()

def plot_connectivity_matrix(matrix, labels=None, title='Connectivity Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
