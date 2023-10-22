
import numpy as np

# Sample dataset (replace this with your own data)

# dataset = np.array([
# [1.0, 2.0, 3.0],
# [2.0, 3.0, 4.0],
# [3.0, 4.0, 5.0],
# ])

dataset = np.array([
[9.2, 2.7, 18.2],
[3.3, 4.7, 11.1],
[3.5, 4.8, 6.9],
])
# Pair of attributes (columns) for which you want to calculate covariance and correlation
attribute1 = 0 # Replace with the index of the first attribute
attribute2 =  1 # Replace with the index of the second attribute
# Calculate covariance between attribute1 and attribute2
covariance = np.cov(dataset[:, attribute1], dataset[:, attribute2])[0, 1]
print(f'Covariance between attribute {attribute1} and attribute {attribute2}: {covariance}')

# Calculate correlation between attribute1 and attribute2
correlation = np.corrcoef(dataset[:, attribute1], dataset[:,
attribute2])[0, 1]
print(f'Correlation between attribute {attribute1} and attribute {attribute2}: {correlation}')
# Extend to compute the Covariance Matrix and Correlation Matrix for the entire dataset
covariance_matrix = np.cov(dataset, rowvar=False)
print('Covariance Matrix:')
print(covariance_matrix)
correlation_matrix = np.corrcoef(dataset, rowvar=False)
print('Correlation Matrix:')
print(correlation_matrix)
