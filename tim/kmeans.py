import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('../data_no_labels.csv')

# Assuming that 'data' is your dataset
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters
kmeans.fit(data)

# Getting the cluster labels
labels = kmeans.predict(data)

# Getting the cluster centers
centers = kmeans.cluster_centers_

print(centers)