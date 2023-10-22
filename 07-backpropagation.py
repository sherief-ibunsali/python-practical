import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define neural network architecture
input_dim = 2
hidden_dim = 4
output_dim = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_dim, hidden_dim))
biases_hidden = np.zeros((1, hidden_dim))
weights_hidden_output = np.random.uniform(size=(hidden_dim, output_dim))
biases_output = np.zeros((1, output_dim))

# Training the neural network using backpropagation
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + biases_output
    output = sigmoid(output_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    biases_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_output = sigmoid(np.dot(sigmoid(np.dot(test_data, weights_input_hidden) + biases_hidden), weights_hidden_output) + biases_output)
predictions = (test_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predictions:")
print(predictions)
