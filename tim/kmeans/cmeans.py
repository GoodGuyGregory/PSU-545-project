import pandas as pd
import numpy as np
import skfuzzy as fuzz

# Load the main dataset
data = pd.read_csv('train.csv')
X = data.iloc[:, :-1].values  # Exclude the label column

# Run Fuzzy C-Means (via the sklearn library)
n_clusters = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Function to compute SSE for a given dataset against each cluster center
def compute_sse(dataset, centers):
    sse = []
    for center in centers:
        distances = np.sum((dataset - center) ** 2, axis=1)
        sse.append(np.sum(distances))
    return sse

# Load each labeled dataset and compute SSE
# Ideally, this would result in each label being assigned to a unique cluster, which would indicate that this method could be used to quickly identify the label of a new data point
labels = ['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc']
for label in labels:
    labeled_data = pd.read_csv(f'{label}.csv').iloc[:, :-1].values  # Exclude the label column
    sse = compute_sse(labeled_data, cntr)
    
    # Determine the best fitting cluster for the labeled dataset
    best_cluster = np.argmin(sse)
    print(f'{label} best fits cluster {best_cluster} with SSE {sse[best_cluster]}')
