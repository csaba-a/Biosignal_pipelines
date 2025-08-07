# Install all required R packages for the biosignal pipeline
packages <- c(
  'readr', 'edfReader', 'tools', 'signal', 'fastICA', 'e1071', 'fractal', 'spectral',
  'RHRV', 'ggplot2', 'corrplot', 'reshape2', 'caret', 'randomForest'
)
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
lapply(packages, install_if_missing) 