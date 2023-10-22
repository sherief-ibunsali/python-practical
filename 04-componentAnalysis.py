import numpy as np

# Sample data matrix (you can replace this with your dataset)
data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])

# Step 1: Standardize the data (mean centering)
mean = np.mean(data, axis=0)
standardized_data = data - mean

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(standardized_data, rowvar=False)

# Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Choose the top k eigenvectors to reduce the dimensionality
k = 2  # You can adjust this based on your desired dimensionality
top_k_eigenvectors = eigenvectors[:, :k]

# Step 6: Project the data onto the top k eigenvectors
reduced_data = np.dot(standardized_data, top_k_eigenvectors)

# Print the reduced data
print("Reduced Dimension Data (top", k, "eigenvectors):")
print(reduced_data)
