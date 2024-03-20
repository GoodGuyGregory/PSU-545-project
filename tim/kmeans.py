import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load data and true labels from CSV files
data_file = 'data.csv'
labels_file = 'labels.csv'
data = pd.read_csv(data_file)
true_labels = pd.read_csv(labels_file, header=None).squeeze()

# Encode string labels to integers for correlation
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Apply K-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
predicted_labels = kmeans.fit_predict(data)

# Create a mapping from predicted labels to true labels
cluster_label_mapping = {}
for cluster in range(kmeans.n_clusters):
    # Find the index of points in each cluster
    indices = [i for i, label in enumerate(predicted_labels) if label == cluster]
    # Get the most common true label for each cluster
    common_true_label = pd.Series(true_labels_encoded[indices]).mode()[0]
    # Map the cluster label to the most common true label
    cluster_label_mapping[cluster] = label_encoder.inverse_transform([common_true_label])[0]

print(cluster_label_mapping)
