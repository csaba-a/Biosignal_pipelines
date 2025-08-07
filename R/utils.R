library(readr)
library(edfReader)
library(tools)

load_csv <- function(file_path) {
  read_csv(file_path)
}

load_edf <- function(file_path) {
  # Returns a list with signal and header info
  edf <- readEdfSignals(file_path)
  return(edf)
}

get_sampling_rate <- function(edf) {
  # Assumes first signal's sampling rate is representative
  edf$signalHeader$samplingrate[1]
}

get_channel_names <- function(edf) {
  edf$signalHeader$label
}

ensure_dir <- function(directory) {
  if (!dir.exists(directory)) dir.create(directory, recursive = TRUE)
}
