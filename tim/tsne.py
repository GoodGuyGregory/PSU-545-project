from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# read in the data file
data = pd.read_csv('../test_data_no_labels.csv')

# read in the label file
labels = pd.read_csv('../test_data_labels.csv')

#tsne
tsne = TSNE(n_components=2, random_state=0)  # n_components is the dimension of the embedded space.

embedded_data = tsne.fit_transform(data)
embedded_df = pd.DataFrame(embedded_data)
embedded_df.to_csv('embedded_data.csv', index=False)

# combined embedded data with labels
embedded_df['label'] = labels
embedded_df.to_csv('embedded_data_with_labels.csv', index=False)


# plots the data, labeling each point with its label and assigning a color to each label
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels)



plt.figure(figsize=(10, 8))




plt.title('t-SNE plot')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('tsne_plot.jpg', dpi=300)  # Saves the plot as a JPG file
plt.show()