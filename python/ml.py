from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def feature_selection(X, y, n_features=10):
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)

def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = SVC(kernel='rbf', probability=True)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
