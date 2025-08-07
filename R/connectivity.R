library(signal)
library(stats)

compute_coherence <- function(sig1, sig2, fs) {
  spec <- mvspec(cbind(sig1, sig2), spans = c(3,3), plot = FALSE)
  f <- spec$freq * fs
  coh <- Mod(spec$fxx) / sqrt(Mod(spec$fxx) * Mod(spec$fyy))
  list(freq = f, coherence = coh)
}

compute_correlation_matrix <- function(data) {
  cor(data, use = "pairwise.complete.obs")
}

compute_phase_locking_value <- function(sig1, sig2) {
  phase1 <- Arg(fft(sig1))
  phase2 <- Arg(fft(sig2))
  plv <- abs(mean(exp(1i * (phase1 - phase2))))
  plv
}
