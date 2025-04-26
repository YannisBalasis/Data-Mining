import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Load Dataset
df = pd.read_csv('data.csv', engine='python', nrows=10000)

# Dataset Information and Description
print("Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nFirst Rows of the Dataset:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# Categorical Variable Analysis
print("\nDistribution of Label:")
print(df['Label'].value_counts())

print("\nDistribution of Traffic Type:")
print(df['Traffic Type'].value_counts())

# Select Numerical Columns
numeric_cols = df.select_dtypes(include='number').columns
desc = df[numeric_cols].describe().T

# Mean per Feature
plt.figure(figsize=(12, 6))
desc['mean'].plot(kind='bar', color='skyblue')
plt.title('Mean Value per Feature')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Standard Deviation per Feature
plt.figure(figsize=(12, 6))
desc['std'].plot(kind='bar', color='orange')
plt.title('Standard Deviation per Feature')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Minimum and Maximum Values per Feature
plt.figure(figsize=(14, 6))
desc[['min', 'max']].plot(kind='bar')
plt.title('Minimum and Maximum Values per Feature')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Combined Statistical Summary
plt.figure(figsize=(16, 8))
desc[['mean', 'std', 'min', '50%', 'max']].plot(kind='bar')
plt.title('Comprehensive Statistical Summary')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.legend(title='Statistics')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
print("\nCorrelation Matrix:")

plt.figure(figsize=(16, 12))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

# ===================================
# Question 2: Dimensionality Reduction and Clustering
# ===================================

# Feature Scaling
print("\nPerforming Dimensionality Reduction using PCA...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create DataFrame with PCA Results
df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['Label'] = df['Label'].values

# Sampling
print("\nCreating a Random Sample of the Dataset...")
df_sample = df.sample(frac=0.2, random_state=42)

# KMeans Clustering
print("\nApplying KMeans Clustering...")
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

df_kmeans = df.copy()
df_kmeans['Cluster_KMeans'] = kmeans_labels

# Agglomerative Clustering
print("\nApplying Agglomerative Clustering on a Sample...")
small_sample_idx = np.random.choice(scaled_data.shape[0], size=10000, replace=False)
scaled_small_sample = scaled_data[small_sample_idx]

agg = AgglomerativeClustering(n_clusters=5)
agg_labels = agg.fit_predict(scaled_small_sample)

df_agg = df.copy()
df_agg['Cluster_Agg'] = agg_labels

# Save New Datasets
print("\nSaving New Subsets...")

df_sample.to_csv("subset_sample.csv", index=False)
df_kmeans.to_csv("subset_kmeans.csv", index=False)
df_agg.to_csv("subset_agg.csv", index=False)

print("Subsets created: subset_sample.csv, subset_kmeans.csv, subset_agg.csv")
