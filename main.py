import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Define the neural network architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784-dimensional vectors
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    layers.Dropout(0.2),                   # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax') # Output layer with 10 units (for 10 classes) and softmax activation
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Step 5: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Step 6: Visualize training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()