# Neural Network for Digit Identification on TMNIST Data

This repository demonstrates the creation of a neural network model for digit identification using the TMNIST (Transposed MNIST) dataset. We have designed a Convolutional Neural Network (CNN) model using the Keras library with the following layers:

## Model Architecture

```python
cnnModel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer with 32 output channels

    Conv2D(32, (3, 3), activation='relu'),  # Convolutional layer with 32 output channels

    MaxPooling2D((2, 2)),  # Max pooling layer to reduce the spatial dimensions

    Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer with 64 output channels

    Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer with 64 output channels

    MaxPooling2D((2, 2)),  # Max pooling layer to reduce the spatial dimensions

    Flatten(),  # Flatten the output for input to fully connected layers

    Dropout(0.5),  # Dropout layer to prevent overfitting

    Dense(512, activation='relu'),  # Another fully connected layer with 512 output units

    Dropout(0.5),  # Dropout layer to prevent overfitting

    Dense(10, activation='softmax')  # Final fully connected layer with 10 output units (corresponding to classes)
])
```

## Model Explanation

- **Convolutional Layers**: We start with two convolutional layers, each with 32 output channels, followed by ReLU activation functions. These layers help extract features from the input images.

- **Max Pooling Layers**: After each pair of convolutional layers, we apply max pooling layers with a (2, 2) pool size to reduce the spatial dimensions.

- **More Convolutional Layers**: We then add two more convolutional layers, each with 64 output channels, followed by ReLU activation functions.

- **Flatten Layer**: To connect the convolutional layers to fully connected layers, we flatten the output.

- **Dropout Layers**: To prevent overfitting, we include dropout layers with a dropout rate of 0.5 after both the first fully connected layer and the second fully connected layer.

- **Fully Connected Layers**: The first fully connected layer consists of 512 output units with ReLU activation, and the final fully connected layer has 10 output units with softmax activation for classification.

## Dataset

We use the TMNIST dataset, which is a transposed version of the MNIST dataset. It contains 28x28 grayscale images of hand-written digits (0-9). The goal is to train the model to classify these digits accurately.
