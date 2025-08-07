library(e1071)
library(fractal)
library(spectral)

extract_time_features <- function(signal) {
  c(
    mean = mean(signal, na.rm = TRUE),
    sd = sd(signal, na.rm = TRUE),
    var = var(signal, na.rm = TRUE),
    min = min(signal, na.rm = TRUE),
    max = max(signal, na.rm = TRUE),
    skewness = e1071::skewness(signal, na.rm = TRUE),
    kurtosis = e1071::kurtosis(signal, na.rm = TRUE),
    rms = sqrt(mean(signal^2, na.rm = TRUE)),
    zero_crossings = sum(diff(sign(signal)) != 0, na.rm = TRUE)
  )
}

extract_freq_features <- function(signal, fs) {
  psd <- spectrum(signal, plot = FALSE, method = "pgram")
  freqs <- psd$freq * fs
  total_power <- sum(psd$spec)
  band_power <- function(fmin, fmax) {
    sum(psd$spec[freqs >= fmin & freqs <= fmax])
  }
  c(
    total_power = total_power,
    delta = band_power(0.5, 4),
    theta = band_power(4, 8),
    alpha = band_power(8, 13),
    beta = band_power(13, 30),
    gamma = band_power(30, 45)
  )
}

extract_nonlinear_features <- function(signal) {
  c(
    hurst = tryCatch(hurstexp(signal)$Hs, error = function(e) NA),
    entropy = tryCatch(spectral::entropy(signal), error = function(e) NA)
  )
}

extract_features_df <- function(df, fs) {
  features <- lapply(df, function(sig) {
    c(
      extract_time_features(sig),
      extract_freq_features(sig, fs),
      extract_nonlinear_features(sig)
    )
  })
  features_df <- as.data.frame(do.call(rbind, features))
  rownames(features_df) <- names(df)
  return(features_df)
}
