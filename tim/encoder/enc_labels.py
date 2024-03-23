import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np

# Load your data and labels
data_file = 'data.csv'
labels_file = 'labels.csv'
data = pd.read_csv(data_file, header=None)
labels = pd.read_csv(labels_file, header=None).squeeze()  # Assuming labels are in a single column

# Encoder and decoder layers configuration
encoder_layer_sizes = [50, 24, 2]  # Example: 3 layers, compressing down to 2 features
decoder_layer_sizes = [8, 16, 24]  # Example: 3 layers, expanding back to original 24 features

# Define the encoder
input_layer = layers.Input(shape=(data.shape[1],))
x = input_layer
for size in encoder_layer_sizes:
    x = layers.Dense(size, activation='relu')(x)
encoded = x

# Define the decoder
for size in decoder_layer_sizes:
    x = layers.Dense(size, activation='relu')(x)
decoded = x

# Build the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Define the encoder model for plotting
encoder = Model(input_layer, encoded)

# Encode the data
encoded_data = encoder.predict(data)

# Plot the encoded data, coloring by labels
unique_labels = labels.unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    indices = labels == label
    plt.scatter(encoded_data[indices, 0], encoded_data[indices, 1], color=color, label=label)

plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.title('2D Encoded data representation with labels')
plt.legend()
plt.savefig('enc_label_plot.jpg', dpi=300)  # Saves the plot as a JPG file
plt.show()
