import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Set working directory
import os
os.chdir("path/to/your/data")

# Function to load biosignal data
def load_biosignal_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function for biosignal data preprocessing
def preprocess_biosignal_data(data):
    # Example: Remove outliers using the interquartile range (IQR)
    def remove_outliers(column):
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        return column[(column >= lower_limit) & (column <= upper_limit)]

    # Apply outlier removal to all numeric columns
    data = data.apply(remove_outliers, axis=0)

    # Impute missing values using median
    data = data.dropna()  # Remove rows with missing values

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    return data

# Function for biosignal data visualization
def visualize_biosignal_data(data):
    # Example: Pair plot for multiple variables with classification
    sns.pairplot(data, vars=['signal1', 'signal2', 'signal3'], hue='label', markers=["o", "s"])

    # Save pair plot
    plt.savefig("pair_plot.png")

# Function for biosignal signal processing
def process_biosignal_data(data):
    # Example: Butterworth bandpass filter
    def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
        nyquist = 0.5 * sampling_rate
        b, a = butter(4, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    # Apply bandpass filter to signal1
    data['filtered_signal1'] = bandpass_filter(data['signal1'], 0.5, 50, 1000)

    return data

# Function for biosignal feature extraction
def extract_biosignal_features(data):
    # Feature extraction using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['signal1', 'signal2', 'signal3']])

    # Feature extraction using FeatureHasher
    hasher = FeatureHasher(n_features=10, input_type='string')
    hashed_features = hasher.transform(data[['feature1', 'feature2']])

    return pd.concat([pd.DataFrame(scaled_data, columns=['scaled_signal1', 'scaled_signal2', 'scaled_signal3']),
                      pd.DataFrame(hashed_features.toarray(), columns=[f'hashed_feature_{i}' for i in range(10)])], axis=1)

# Function for Random Forest classification
def random_forest_classification(train_data, test_data):
    # Machine Learning - Random Forest Classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_data.drop('label', axis=1), train_data['label'])
    predictions = model.predict(test_data.drop('label', axis=1))

    # Evaluate model performance
    conf_matrix = confusion_matrix(test_data['label'], predictions)
    classification_rep = classification_report(test_data['label'], predictions)
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_rep)

    # Save model
    joblib.dump(model, "random_forest_model.joblib")

    return predictions

# Main Script
# Load biosignal data
biosignal_data = load_biosignal_data("your_data.csv")

# Preprocess biosignal data
preprocessed_data = preprocess_biosignal_data(biosignal_data)

# Visualize biosignal data
visualize_biosignal_data(preprocessed_data)

# Process biosignal data
processed_data = process_biosignal_data(preprocessed_data)

# Feature extraction
features = extract_biosignal_features(processed_data)

# Split data for training and testing
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Random Forest Classification
predictions = random_forest_classification(train_data, test_data)
