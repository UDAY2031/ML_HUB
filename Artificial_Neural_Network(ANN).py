import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Data for XOR problem
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # Output

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.uniform(size=(2, 4))  # Input to hidden weights
w2 = np.random.uniform(size=(4, 1))  # Hidden to output weights
b1 = np.random.uniform(size=(1, 4))  # Hidden biases
b2 = np.random.uniform(size=(1, 1))  # Output biases

# Training parameters
lr = 0.5  # Learning rate
epochs = 5000
print_epochs = [0, 100, 200, 300, 400]  # Epochs to display

# Training
for epoch in range(epochs + 1):
    # Forward pass
    h_input = np.dot(x, w1) + b1  # Hidden layer input
    h_output = sigmoid(h_input)   # Hidden layer output
    o_input = np.dot(h_output, w2) + b2  # Output layer input
    o_output = sigmoid(o_input)         # Output layer output

    # Loss
    loss = np.mean((y - o_output) ** 2)

    # Backpropagation
    o_error = y - o_output
    o_delta = o_error * sigmoid_derivative(o_output)

    h_error = o_delta.dot(w2.T)
    h_delta = h_error * sigmoid_derivative(h_output)

    # Update weights and biases
    w2 += h_output.T.dot(o_delta) * lr
    w1 += x.T.dot(h_delta) * lr
    b2 += np.sum(o_delta, axis=0, keepdims=True) * lr
    b1 += np.sum(h_delta, axis=0, keepdims=True) * lr

    # Print loss for specific epochs
    if epoch in print_epochs:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing
o_output_binary = np.round(o_output)  # Convert to binary (0 or 1)
accuracy = np.mean(o_output_binary == y) * 100  # Accuracy

# Final results
print(f"Test Accuracy: {accuracy:.2f}%")
