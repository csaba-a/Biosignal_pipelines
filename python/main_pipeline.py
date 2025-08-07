"""
Main Pipeline for EEG/ECG Analysis (Python)
===========================================

This script runs a full biosignal analysis pipeline for EEG and ECG data.
It supports both CSV and EDF input formats and performs preprocessing, feature extraction,
HRV and connectivity analysis, visualization, and machine learning classification.

---

Requirements:
- Python 3.8+
- mne
- neurokit2
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- antropy (optional, for nonlinear features)

Install requirements:
    pip install mne neurokit2 pandas numpy scikit-learn matplotlib seaborn antropy

Usage:
- Place your data file in the ../data/ directory (CSV or EDF)
- Set the data_path variable below to your file
- Run: python main_pipeline.py
- Results and plots will be saved in ../output/

---
"""

import os
import sys
# Import all pipeline modules
from utils import load_csv, load_edf, ensure_dir
from preprocessing import bandpass_filter, notch_filter, run_ica, apply_ica, normalize_signals, resample_raw, interpolate_bads
from features import extract_features_df
from hrv import compute_hrv
from connectivity import compute_correlation_matrix, compute_coherence, compute_phase_locking_value
from visualization import plot_raw_signal, plot_psd, plot_hrv, plot_connectivity_matrix
from ml import feature_selection, train_random_forest, train_svm, evaluate_model, plot_feature_importance
import mne
import pandas as pd

# --- Config ---
# Set your data file here (CSV or EDF)
data_path = '../data/your_data.edf'  # or 'your_data.csv'
output_dir = '../output/'
ensure_dir(output_dir)

# --- Load Data ---
# Load either EDF (EEG/ECG) or CSV (tabular signals)
if data_path.endswith('.edf'):
    raw = load_edf(data_path)
    sfreq = raw.info['sfreq']
    print(f"Loaded EDF: {data_path}, {len(raw.ch_names)} channels, {sfreq} Hz")
    plot_raw_signal(raw)
else:
    df = load_csv(data_path)
    print(f"Loaded CSV: {data_path}, shape: {df.shape}")
    # For CSV, assume columns are signals, last column is label if present
    sfreq = 1000  # Set manually or infer from metadata

# --- Preprocessing ---
# Filtering, artifact removal, normalization
if data_path.endswith('.edf'):
    raw = bandpass_filter(raw, 0.5, 45)
    raw = notch_filter(raw, freqs=[50, 60])
    # ICA for artifact removal (optional, uncomment to use)
    # ica = run_ica(raw)
    # ica.plot_components()  # Uncomment for manual inspection
    # raw = apply_ica(raw, ica, exclude=[0])  # Example: exclude component 0
    # Interpolate bads (example: ['Fp1'])
    # raw = interpolate_bads(raw, bads=['Fp1'])
    plot_psd(raw)
    # Convert to DataFrame for feature extraction
    df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
else:
    # For CSV, apply normalization
    df.iloc[:, :-1] = normalize_signals(df.iloc[:, :-1])

# --- Feature Extraction ---
# Extract time, frequency, and nonlinear features
features = extract_features_df(df, sfreq)
features_df = pd.DataFrame(features).T
features_df.to_csv(os.path.join(output_dir, 'features.csv'))

# --- HRV (if ECG present) ---
# Computes HRV features if ECG channel is present
if 'ECG' in df.columns or 'ecg' in df.columns:
    ecg_col = 'ECG' if 'ECG' in df.columns else 'ecg'
    hrv_df = compute_hrv(df[ecg_col].values, sampling_rate=sfreq)
    hrv_df.to_csv(os.path.join(output_dir, 'hrv.csv'))
    plot_hrv(hrv_df)

# --- Connectivity (if EEG) ---
# Computes correlation and coherence between channels
if data_path.endswith('.edf'):
    corr_matrix = compute_correlation_matrix(raw)
    plot_connectivity_matrix(corr_matrix, labels=raw.ch_names)
    # Example: coherence between Fp1 and Fp2
    if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
        f, coh = compute_coherence(raw, 'Fp1', 'Fp2')
        # Plot coherence (optional)

# --- Machine Learning ---
# Feature selection, train/test split, model training and evaluation
if 'label' in df.columns:
    X = features_df.values
    y = df['label'].values
    X_new, selected = feature_selection(X, y, n_features=10)
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    # Train and evaluate Random Forest
    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test)
    plot_feature_importance(rf, feature_names=features_df.index[selected])
    # Train and evaluate SVM
    svm = train_svm(X_train, y_train)
    evaluate_model(svm, X_test, y_test)

print('Pipeline complete. Results saved to output/.')
