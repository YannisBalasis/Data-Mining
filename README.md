# Data-Mining
Analysis, dimensionality reduction, and classification on the TII-SSRC-23 network traffic dataset (Intrusion Detection Systems research).


Analysis and Modeling of the TII-SSRC-23 Dataset
Description
This project focuses on the analysis and modeling of the TII-SSRC-23 dataset, which contains network traffic data collected for the development and research of Intrusion Detection Systems (IDS).
The data processing pipeline is divided into two main stages:

Analysis and dimensionality reduction of the dataset

Training and evaluation of classifiers based on Support Vector Machines (SVM) and Neural Networks

The entire implementation was developed in Python.

Project Structure
analysis_and_reduction.py:
Performs dataset analysis, statistical description, visualizations, dimensionality reduction (PCA), and clustering.

train_and_evaluate.py:
Handles the training and evaluation of SVM and Neural Network classifiers on the generated dataset versions.

Detailed Workflow
Question 1 - Data Analysis
Loading and exploring the dataset (data.csv).

Calculation of basic statistical metrics (mean, standard deviation, minimum, maximum).

Visualizations:

Bar plots for mean, standard deviation, minimum, maximum

Heatmap for feature correlation analysis

Exploration of categorical variables: Label and Traffic Type.

Question 2 - Data Reduction
PCA: Dimensionality reduction to 2 principal components.

Sampling: Creation of a random subset representing 20% of the original dataset.

Clustering:

KMeans clustering on the full dataset.

Agglomerative Clustering on a random sample of 10,000 rows.

Creation of three new representative datasets:

subset_sample.csv

subset_kmeans.csv

subset_agg.csv

Question 3 - Model Training
For each newly created dataset:

Training:

An SVM classifier.

A Neural Network classifier.

Prediction targets:

Label (Normal vs Malicious Traffic — Binary Classification)

Traffic Type (Type of Traffic — Multi-Class Classification)

Evaluation metrics:

Accuracy

Precision

Recall

F1-Score

Comprehensive Classification Reports

Libraries Used
pandas, numpy (data processing)

matplotlib, seaborn (data visualization)

scikit-learn (preprocessing, PCA, clustering, SVM, evaluation metrics)

tensorflow/keras (Neural Network modeling)

Dataset
The original dataset can be found on Kaggle. (https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23)
