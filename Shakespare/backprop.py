import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where( x > 0, 1, 0)

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid

# Leaky ReLU activation function and its derivative
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.normal(0, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, 1, (hidden_size, output_size))
        
        # Bias terms
        self.bias_hidden = np.random.normal(0, 1, (1, hidden_size))
        self.bias_output = np.random.normal(0, 1, (1, output_size))
    
    def feedforward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # self.hidden_output = sigmoid(self.hidden_input)
        self.hidden_output = leaky_relu(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # self.output_output = sigmoid(self.output_input)
        self.output_output = leaky_relu(self.output_input)
        
        return self.output_output
    
    def backpropagate(self, X, y, learning_rate):
        # Forward pass
        output = self.feedforward(X)
        
        # Calculate error (difference between actual and predicted output)
        error_output = y - output
        
        # Calculate gradients for the output layer
        # d_output = error_output * sigmoid_derivative(output)
        d_output = error_output * leaky_relu_derivative(output)
        
        # Calculate error for hidden layer by propagating backward
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        
        # Calculate gradients for the hidden layer
        # d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        d_hidden = error_hidden * leaky_relu_derivative(self.hidden_output)
        
        # Update the weights and biases (using gradient descent)
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagate(X, y, learning_rate)
            
            # Optionally print error at certain intervals
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.feedforward(X)))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage with sample data
if __name__ == "__main__":
    # Input data (4 examples with 2 input features)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    # Target output (XOR problem)
    y = np.array([[0], [1], [1], [0]])
    
    # Initialize neural network (2 input neurons, 2 hidden neurons, 1 output neuron)
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    
    # Train the neural network
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Test the network after training
    print("Output after training:")
    print(nn.feedforward(X))
