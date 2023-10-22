from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the k-NN classifier
k = 3  # You can change the value of k as needed
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier on the training data
knn.fit(X_train, y_train)

# Predict the classes for the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0

for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Correct Prediction: Actual class {y_test[i]}, Predicted class {y_pred[i]}")
        correct_predictions += 1
    else:
        print(f"Wrong Prediction: Actual class {y_test[i]}, Predicted class {y_pred[i]}")
        wrong_predictions += 1

print(f"Total Correct Predictions: {correct_predictions}")
print(f"Total Wrong Predictions: {wrong_predictions}")
