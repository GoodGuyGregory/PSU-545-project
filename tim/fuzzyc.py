import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import LabelEncoder

# Load data and true labels from CSV files
data_file = 'data.csv'
labels_file = 'labels.csv'
data = pd.read_csv(data_file).values  # Ensure data is in numpy array format for skfuzzy
true_labels = pd.read_csv(labels_file, header=None).squeeze()

# Encode string labels to integers for correlation
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Apply Fuzzy C-Means clustering with 5 clusters
n_clusters = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Find the cluster number for each data point
labels_fcm = np.argmax(u, axis=0)

# Create a mapping from FCM cluster numbers to true labels
cluster_label_mapping = {}
for cluster in range(n_clusters):
    # Find the indices of points in each cluster
    indices = np.where(labels_fcm == cluster)[0]
    # Get the most common true label for each cluster
    if indices.size > 0:
        common_true_label = pd.Series(true_labels_encoded[indices]).mode()[0]
        # Map the cluster number to the most common true label
        cluster_label_mapping[cluster] = label_encoder.inverse_transform([common_true_label])[0]

print(cluster_label_mapping)
