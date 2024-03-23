import pandas as pd
from sklearn.cluster import KMeans

# Load the main dataset
data = pd.read_csv('train.csv')
X = data.iloc[:, :-1]  # Exclude the label column

# Run k-means clustering (via the sklearn library)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
clusters = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Function to compute SSE for a given dataset against each cluster center
# used to check the accuracy of a cluster assignment for a particular label
def compute_sse(dataset, centers):
    sse = []
    for center in centers:
        distances = ((dataset - center) ** 2).sum(axis=1)
        sse.append(distances.sum())
    return sse

# Load each labeled dataset and compute SSE
# Ideally, this would result in each label being assigned to a unique cluster, which would indicate that this method could be used to quickly identify the label of a new data point
labels = ['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc']
for label in labels:
    labeled_data = pd.read_csv(f'{label}.csv').iloc[:, :-1]  # Exclude the label column
    sse = compute_sse(labeled_data, centers)
    
    # Determine the best fitting cluster for the labeled dataset
    best_cluster = min(range(len(sse)), key=lambda k: sse[k])
    print(f'{label} best fits cluster {best_cluster} with SSE {sse[best_cluster]}')
