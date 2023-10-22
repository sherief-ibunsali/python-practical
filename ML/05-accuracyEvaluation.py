import numpy as np

# Sample training data stored as arrays (you can replace this with your data)
X_train = np.array([
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
])
y_train = np.array([1, 0, 1, 0, 1, 0])  # Labels

# Sample test data stored as arrays (you can replace this with your data)
X_test = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
])
y_test = np.array([0, 1, 1])  # True labels for test data

# Function to calculate class probabilities and make predictions
def naive_bayes_classifier(X_train, y_train, X_test):
    # Calculate class probabilities
    total_samples = len(y_train)
    num_features = X_train.shape[1]
    unique_classes = np.unique(y_train)
    class_probs = {}
    predictions = []

    for class_label in unique_classes:
        class_mask = (y_train == class_label)
        class_probs[class_label] = sum(class_mask) / total_samples
        for feature in range(num_features):
            class_mask_feature = X_train[y_train == class_label]
            feature_probs = class_mask_feature[:, feature]
            class_probs[class_label] *= (sum(feature_probs) + 1) / (sum(class_mask) + 2)

    # Make predictions
    for sample in X_test:
        sample_probs = {}
        for class_label in unique_classes:
            sample_prob = class_probs[class_label]
            for feature_idx, feature_value in enumerate(sample):
                sample_prob *= (X_train[y_train == class_label][:, feature_idx] == feature_value).sum() / sum(y_train == class_label)
            sample_probs[class_label] = sample_prob
        predictions.append(max(sample_probs, key=sample_probs.get))

    return predictions

# Use the Naive Bayesian classifier
predicted_labels = naive_bayes_classifier(X_train, y_train, X_test)

# Calculate accuracy
accuracy = (predicted_labels == y_test).mean()
print(f"Accuracy: {accuracy:.2f}")
