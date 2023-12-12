import numpy as np
import pandas as pd

# Create a 2D dataset
np.random.seed(42)
X_2d = 2 * np.random.rand(100, 1)
y_2d = 4 + 3 * X_2d + np.random.randn(100, 1)

# Save the dataset to a CSV file
dataset_2d = pd.DataFrame(np.hstack([X_2d, y_2d]), columns=['Feature1', 'Target'])
dataset_2d.to_csv('examples/dataset_2d.csv', index=False)

# Create a 3D dataset
X1_3d = 2 * np.random.rand(100, 1)
X2_3d = 3 * np.random.rand(100, 1)
y_3d = 4 + 3 * X1_3d + 2 * X2_3d + np.random.randn(100, 1)

# Save the dataset to a CSV file
dataset_3d = pd.DataFrame(np.hstack([X1_3d, X2_3d, y_3d]), columns=['Feature1', 'Feature2', 'Target'])
dataset_3d.to_csv('examples/dataset_3d.csv', index=False)
