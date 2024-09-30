import numpy as np

def relu(x):
    return max( 0, x)

def perceptron_learning(examples, learning_rate=0.01, max_epochs=1000, norm_threshold=1.0):
    # Initialize weights and bias (including the bias as an extra weight w0)
    n_features = examples[0][0].shape[0]  # Number of features from the first example input
    # weights = np.zeros(n_features)  # Initialize weights as zeroes
    weights = np.random.normal(loc=0.0, scale=1.0, size=n_features) # initalise the weighs as gaussian
    for epoch in range(max_epochs):
        for x, y in examples:  # Iterate through each example
            # Compute the input to the activation function
            in_val = np.dot(weights, x)  # Summing up weights * inputs

            # Activation function, here assuming a step function
            # y_hat = 1 if in_val >= 0 else 0  # Step function for perceptron
            y_hat = relu(in_val)

            # Compute the error
            err = y - y_hat

            # Update the weights
            weights += learning_rate * err * x

            # Normalize the weights if their L2 norm exceeds the threshold
            weight_norm = np.linalg.norm(weights)
            if weight_norm > norm_threshold:
                weights = weights / weight_norm * norm_threshold  # Rescale the weights

            if epoch % 100 == 0:
                print(f"Epochs {epoch}, error {err}")
        # Optionally, you can add a stopping criterion here if error becomes sufficiently small

    return weights

# Example usage with sample data
if __name__ == "__main__":
    # Define example inputs (each x is a vector of features, y is the target output)
    examples = [
        (np.array([1.0, 0]), 1.0),  # Example 1
        (np.array([0, 1.0]), 0),  # Example 2
        (np.array([1, 1.0]), 1.0),  # Example 3
        (np.array([0, 0]), 0)   # Example 4
    ]
    
    weights = perceptron_learning(examples)
    print("Learned weights:", weights)
