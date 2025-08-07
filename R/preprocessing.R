library(signal)
library(fastICA)

bandpass_filter <- function(signal, low, high, fs) {
  bf <- butter(4, c(low, high) / (fs / 2), type = "pass")
  filtfilt(bf, signal)
}

notch_filter <- function(signal, freq, fs) {
  w0 <- freq / (fs / 2)
  bw <- w0 / 35
  nf <- signal::butter(2, c(w0 - bw, w0 + bw), type = "stop")
  filtfilt(nf, signal)
}

run_ica <- function(data, n.comp = 15) {
  fastICA::fastICA(data, n.comp)
}

apply_ica <- function(data, ica, exclude = NULL) {
  if (!is.null(exclude)) {
    ica$S[, exclude] <- 0
    return(ica$A %*% t(ica$S))
  }
  return(ica$A %*% t(ica$S))
}

normalize_signals <- function(data) {
  scale(data)
}

resample_signal <- function(signal, orig_fs, new_fs) {
  n <- length(signal)
  t_orig <- seq(0, (n - 1) / orig_fs, by = 1 / orig_fs)
  t_new <- seq(0, max(t_orig), by = 1 / new_fs)
  approx(t_orig, signal, xout = t_new)$y
}

interpolate_bads <- function(data, bads) {
  for (bad in bads) {
    data[, bad] <- zoo::na.approx(data[, bad], na.rm = FALSE)
  }
  return(data)
}
