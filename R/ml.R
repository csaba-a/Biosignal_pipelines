library(caret)
library(randomForest)
library(e1071)

feature_selection <- function(X, y, n_features = 10) {
  ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
  results <- rfe(X, y, sizes = n_features, rfeControl = ctrl)
  list(selected = predictors(results), results = results)
}

train_random_forest <- function(X, y) {
  randomForest(x = X, y = as.factor(y), ntree = 100)
}

train_svm <- function(X, y) {
  svm(x = X, y = as.factor(y), kernel = "radial", probability = TRUE)
}

evaluate_model <- function(model, X_test, y_test) {
  preds <- predict(model, X_test)
  print(confusionMatrix(preds, as.factor(y_test)))
}

plot_feature_importance <- function(model, feature_names) {
  if (!is.null(model$importance)) {
    importance <- model$importance
    barplot(importance, names.arg = feature_names, las = 2, main = "Feature Importance")
  }
}
