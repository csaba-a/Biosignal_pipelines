library(ggplot2)
library(corrplot)
library(reshape2)

plot_raw_signal <- function(df, channels = NULL, start = 1, end = 1000) {
  if (is.null(channels)) channels <- names(df)[1:5]
  df_long <- melt(df[start:end, channels], variable.name = "Channel", value.name = "Value")
  df_long$Time <- rep(start:end, length(channels))
  ggplot(df_long, aes(x = Time, y = Value, color = Channel)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Raw Signals", x = "Time", y = "Amplitude")
}

plot_psd <- function(signal, fs) {
  psd <- spectrum(signal, plot = FALSE)
  df <- data.frame(Frequency = psd$freq * fs, Power = psd$spec)
  ggplot(df, aes(x = Frequency, y = Power)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Power Spectral Density", x = "Frequency (Hz)", y = "Power")
}

plot_hrv <- function(hrv_df) {
  ggplot(hrv_df, aes(x = Time, y = MeanNN)) +
    geom_line() +
    theme_minimal() +
    labs(title = "HRV MeanNN Over Time", x = "Time", y = "MeanNN (ms)")
}

plot_connectivity_matrix <- function(matrix, labels = NULL, title = "Connectivity Matrix") {
  corrplot(matrix, method = "color", tl.col = "black", title = title)
}
