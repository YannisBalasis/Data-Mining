# model_training_and_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Function for Training and Evaluating Models
def train_and_evaluate_models(df, target_column, description):
    X = df.drop(columns=[target_column, 'Traffic Subtype', 'Traffic Type', 'Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')
    y = df[target_column]

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # SVM Classifier
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)

    # Neural Network Classifier
    num_classes = len(np.unique(y))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    if num_classes > 2:
        y_pred_nn = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred_nn = (model.predict(X_test) > 0.5).astype("int32").flatten()

    nn_acc = accuracy_score(y_test, y_pred_nn)

    return svm_acc, nn_acc

# Load the Datasets
df_sample = pd.read_csv('subset_sample.csv')
df_kmeans = pd.read_csv('subset_kmeans.csv')
df_agg = pd.read_csv('subset_agg.csv')

subsets = ['Sample', 'KMeans', 'Agglomerative']
subset_dfs = [df_sample, df_kmeans, df_agg]

# Lists to Store Results
svm_label_acc = []
nn_label_acc = []
svm_traffic_acc = []
nn_traffic_acc = []

# Train Models for Label Prediction (Binary Classification)
for df in subset_dfs:
    svm_acc, nn_acc = train_and_evaluate_models(df, target_column='Label', description='Label')
    svm_label_acc.append(svm_acc)
    nn_label_acc.append(nn_acc)

# Train Models for Traffic Type Prediction (Multi-Class Classification)
for df in subset_dfs:
    svm_acc, nn_acc = train_and_evaluate_models(df, target_column='Traffic Type', description='Traffic Type')
    svm_traffic_acc.append(svm_acc)
    nn_traffic_acc.append(nn_acc)

# Visualization of Accuracy Scores

x = np.arange(len(subsets))
width = 0.35

# Bar Plot for Label Prediction Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, svm_label_acc, width, label='SVM')
ax.bar(x + width/2, nn_label_acc, width, label='Neural Network')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Label Prediction (Benign vs Malicious)')
ax.set_xticks(x)
ax.set_xticklabels(subsets)
ax.set_ylim(0.98, 1.01)
ax.legend()

for bar in ax.containers:
    ax.bar_label(bar, fmt='%.4f', label_type='edge')

plt.tight_layout()
plt.show()

# Bar Plot for Traffic Type Prediction Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, svm_traffic_acc, width, label='SVM')
ax.bar(x + width/2, nn_traffic_acc, width, label='Neural Network')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Traffic Type Prediction')
ax.set_xticks(x)
ax.set_xticklabels(subsets)
ax.set_ylim(0.98, 1.01)
ax.legend()

for bar in ax.containers:
    ax.bar_label(bar, fmt='%.4f', label_type='edge')

plt.tight_layout()
plt.show()
