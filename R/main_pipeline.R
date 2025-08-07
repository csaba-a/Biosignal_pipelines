# Main Pipeline for EEG/ECG Analysis (R)
# ======================================
#
# This script runs a full biosignal analysis pipeline for EEG and ECG data.
# It supports both CSV and EDF input formats and performs preprocessing, feature extraction,
# HRV and connectivity analysis, visualization, and machine learning classification.
#
# ---
#
# Requirements:
# - R >= 4.0
# - Packages: readr, edfReader, tools, signal, fastICA, e1071, fractal, spectral, RHRV, ggplot2, corrplot, reshape2, caret, randomForest, e1071
#
# Install requirements (in R):
#   install.packages(c('readr', 'edfReader', 'tools', 'signal', 'fastICA', 'e1071', 'fractal', 'spectral', 'RHRV', 'ggplot2', 'corrplot', 'reshape2', 'caret', 'randomForest'))
#
# Usage:
# - Place your data file in the ../data/ directory (CSV or EDF)
# - Set the data_path variable below to your file
# - Run: source('main_pipeline.R')
# - Results and plots will be saved in ../output/
#
# ---

# Load all pipeline modules
source('utils.R')
source('preprocessing.R')
source('features.R')
source('hrv.R')
source('connectivity.R')
source('visualization.R')
source('ml.R')

# --- Config ---
# Set your data file here (CSV or EDF)
data_path <- '../data/your_data.edf' # or 'your_data.csv'
output_dir <- '../output/'
ensure_dir(output_dir)

# --- Load Data ---
# Load either EDF (EEG/ECG) or CSV (tabular signals)
if (file_ext(data_path) == 'edf') {
  edf <- load_edf(data_path)
  fs <- get_sampling_rate(edf)
  cat(sprintf('Loaded EDF: %s, %d channels, %.1f Hz\n', data_path, length(get_channel_names(edf)), fs))
  df <- as.data.frame(edf$signalMatrix)
  plot_raw_signal(df)
} else {
  df <- load_csv(data_path)
  cat(sprintf('Loaded CSV: %s, shape: %d x %d\n', data_path, nrow(df), ncol(df)))
  fs <- 1000 # Set manually or infer from metadata
}

# --- Preprocessing ---
# Filtering, artifact removal, normalization
if (file_ext(data_path) == 'edf') {
  for (ch in names(df)) {
    df[[ch]] <- bandpass_filter(df[[ch]], 0.5, 45, fs)
    df[[ch]] <- notch_filter(df[[ch]], 50, fs)
  }
  # ICA for artifact removal (optional, uncomment to use)
  # ica <- run_ica(as.matrix(df), n.comp = min(15, ncol(df)))
  # df <- as.data.frame(apply_ica(as.matrix(df), ica, exclude = c(1)))
  plot_psd(df[[1]], fs)
}

# --- Feature Extraction ---
# Extract time, frequency, and nonlinear features
features_df <- extract_features_df(df, fs)
write.csv(features_df, file.path(output_dir, 'features.csv'))

# --- HRV (if ECG present) ---
# Computes HRV features if ECG channel is present
if ('ECG' %in% names(df) || 'ecg' %in% names(df)) {
  ecg_col <- if ('ECG' %in% names(df)) 'ECG' else 'ecg'
  hrv_list <- compute_hrv(df[[ecg_col]], sampling_rate = fs)
  write.csv(hrv_list$time, file.path(output_dir, 'hrv_time.csv'))
  write.csv(hrv_list$freq, file.path(output_dir, 'hrv_freq.csv'))
  # plot_hrv(hrv_list$time) # Uncomment if you want to plot
}

# --- Connectivity (if EEG) ---
# Computes correlation and coherence between channels
if (file_ext(data_path) == 'edf') {
  corr_matrix <- compute_correlation_matrix(df)
  plot_connectivity_matrix(corr_matrix, labels = names(df))
  # Example: coherence between first two channels
  if (ncol(df) >= 2) {
    coh <- compute_coherence(df[[1]], df[[2]], fs)
    # Plot coherence (optional)
  }
}

# --- Machine Learning ---
# Feature selection, train/test split, model training and evaluation
if ('label' %in% names(df)) {
  X <- as.matrix(features_df)
  y <- as.factor(df$label)
  fs_res <- feature_selection(X, y, n_features = 10)
  selected <- fs_res$selected
  X_new <- X[, selected, drop = FALSE]
  set.seed(42)
  train_idx <- sample(seq_len(nrow(X_new)), size = 0.8 * nrow(X_new))
  X_train <- X_new[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X_new[-train_idx, ]
  y_test <- y[-train_idx]
  rf <- train_random_forest(X_train, y_train)
  evaluate_model(rf, X_test, y_test)
  plot_feature_importance(rf, feature_names = selected)
  svm_model <- train_svm(X_train, y_train)
  evaluate_model(svm_model, X_test, y_test)
}

cat('Pipeline complete. Results saved to output/.\n')
