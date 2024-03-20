import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load data and labels from CSV files
data_file = '../test_data_no_labels.csv'
labels_file = '../test_data_labels.csv'
data = pd.read_csv(data_file, header=None)
labels = pd.read_csv(labels_file, header=None).iloc[:, 0]  # Make sure labels are read as a series

# Check if data and labels have the same length
if len(data) != len(labels):
    raise ValueError("Data and labels files must have the same number of rows.")

# Apply t-SNE on the data
tsne = TSNE(n_components=2, random_state=0)
data_embedded = tsne.fit_transform(data)

# Plot the results with labels
plt.figure(figsize=(10, 8))
unique_labels = labels.unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    indices = labels == label
    plt.scatter(data_embedded[indices, 0], data_embedded[indices, 1], color=color, label=label)

plt.title('t-SNE plot with labels')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

# Save the plot as a JPG file
plt.savefig('tsne_plot.jpg', format='jpg')
