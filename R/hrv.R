library(RHRV)

compute_hrv <- function(ecg_signal, sampling_rate) {
  hrv.data <- CreateHRVData()
  hrv.data <- SetVerbose(hrv.data, FALSE)
  hrv.data <- LoadBeatVector(hrv.data, which(ecg_signal == 1) / sampling_rate)
  hrv.data <- BuildNIHR(hrv.data)
  hrv.data <- FilterNIHR(hrv.data)
  hrv.data <- InterpolateNIHR(hrv.data, freqhr = 4)
  hrv.data <- CreateTimeAnalysis(hrv.data, size = 300, interval = 7.8125)
  hrv.data <- CreateFreqAnalysis(hrv.data)
  hrv.data <- CalculatePowerBand(hrv.data, indexFreqAnalysis = 1, size = 300, shift = 30, type = "fourier")
  time_features <- hrv.data$TimeAnalysis
  freq_features <- hrv.data$FreqAnalysis[[1]]$Bands
  list(time = time_features, freq = freq_features)
}
