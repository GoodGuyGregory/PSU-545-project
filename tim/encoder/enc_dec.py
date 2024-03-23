import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Load your data
data_file = 'data.csv'
data = pd.read_csv(data_file)


# Define the encoder part
input_dim = 24  # Number of features in the input data
encoding_dim = 2  # Compressed representation size

# Encoder and decoder layers configuration
encoder_layer_sizes = [50, 50, 24, 2]  # Example: 3 layers, compressing down to 2 features
decoder_layer_sizes = [8, 12, 16, 20, 24]  # Example: 3 layers, expanding back to original 24 features

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



# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Define the encoder model
encoder = Model(input_layer, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Use the encoder to get the compressed representation of the data
encoded_data = encoder.predict(data)

print(encoded_data.shape)
# Plot the encoded data
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.title('2D Encoded data representation')
plt.savefig('enc_plot.jpg', dpi=300)  # Saves the plot as a JPG file
plt.show()
