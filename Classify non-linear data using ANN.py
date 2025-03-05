import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)
    
    def backward(self, X, y):
        m = X.shape[0]
        output = self.forward(X)
        error = output - y
        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m
        hidden_error = np.dot(error, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch)
            
            if epoch % 10 == 0:
                output = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

def main():
    # Load dataset
    data_path = 'nonLinear_data.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X, y = data['X'], data['labels']

    # One-hot encoding
    y_one_hot = np.eye(len(set(y.flatten())))[y.flatten()]

    # Split dataset
    split_ratio = 0.8
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * split_ratio)
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_one_hot[train_indices], y_one_hot[test_indices]

    # Train model
    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=8, output_size=y_one_hot.shape[1], learning_rate=0.1)
    nn.train(X_train, y_train, epochs=100, batch_size=32)

    # Predict
    y_pred = nn.predict(X_test)
    print('Predictions on test set:', y_pred)

if __name__ == "__main__":
    main()
