import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def add_bias(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))

def predict(w, X):
    return (np.dot(X, w) >= 0).astype(int)

def train_perceptron(X, y, max_iterations=100):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    iterations = 0
    completed = False

    while not completed and iterations < max_iterations:
        completed = True
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]
            prediction = 1 if np.dot(w, xi) >= 0 else 0

            # If misclassified, update the weights according to the rule:
            # If yi = 1 and prediction = 0, update w = w + xi.
            # If yi = 0 and prediction = 1, update w = w - xi.
            if yi == 1 and prediction == 0:
                w = w + xi
                completed = False
            elif yi == 0 and prediction == 1:
                w = w - xi
                completed = False
        iterations += 1
    return w, iterations

def plot_decision_boundary(ax, w, xlim, label='Decision Boundary'):
    x_vals = np.linspace(xlim[0], xlim[1], 200)
    # Avoid division by zero if w[1] is near 0
    if np.abs(w[1]) > 1e-6:
        y_vals = -(w[0] * x_vals + w[2]) / w[1]
        ax.plot(x_vals, y_vals, 'k-', label=label)
    else:
        # Vertical line: x = -bias/w[0]
        x_line = -w[2] / w[0]
        ax.axvline(x=x_line, color='k', label=label)

def main():
    training_files = [f"twoclassData/set{i}.train" for i in range(1, 11)]
    test_file = "twoclassData/set.test"

    X_test, y_test = load_data(test_file)
    X_test_bias = add_bias(X_test)

    error_rates = []
    epoch_counts = []

    for i, train_file in enumerate(training_files, start=1):
        # Load training data and add bias
        X_train, y_train = load_data(train_file)
        X_train_bias = add_bias(X_train)

        # Train the perceptron on the current training set
        w, iterations = train_perceptron(X_train_bias, y_train)
        epoch_counts.append(iterations)

        # Evaluate misclassification error on the test set
        predictions = predict(w, X_test_bias)
        error_rate = np.mean(predictions != y_test)
        error_rates.append(error_rate)

        # Plot training data with decision boundary
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Left: training set
        ax1 = axes[0]
        ax1.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='red', marker='o', label='Class 0')
        ax1.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='blue', marker='x', label='Class 1')
        # Determine x limits based on training data
        xlim = [np.min(X_train[:,0]) - 1, np.max(X_train[:,0]) + 1]
        plot_decision_boundary(ax1, w, xlim)
        ax1.set_title(f"Training Set {i}\nIterations: {iterations}")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.legend()

        # Right: test set
        ax2 = axes[1]
        ax2.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], color='red', marker='o', alpha=0.5, label='Class 0')
        ax2.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], color='blue', marker='x', alpha=0.5, label='Class 1')
        # Use x limits from the test set for consistency
        xlim_test = [np.min(X_test[:,0]) - 1, np.max(X_test[:,0]) + 1]
        plot_decision_boundary(ax2, w, xlim_test)
        ax2.set_title(f"Test Set\nMisclassification Error: {error_rate:.3f}")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.legend()

        plt.tight_layout()
        # Save the figure for the current training set
        plt.savefig(f"plots/perceptron_set{i}.png")
        plt.show()

        # print(f"Training file: {train_file} | Iterations: {iterations} | Test error rate: {error_rate:.3f}")

    # Summary of results
    print("\nSummary of Training:")
    for i, (ep, err) in enumerate(zip(epoch_counts, error_rates), start=1):
        print(f"Set {i}: iterations = {ep}, Test Error Rate = {err:.3f}")

if __name__ == "__main__":
    main()