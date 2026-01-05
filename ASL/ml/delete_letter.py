import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

# Load data
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Current dataset size:", X.shape, y.shape)
print("A=0, B=1, C=2, ... Z=25")

label = int(input("Enter label number to DELETE: "))

# Create mask to keep all other labels
mask = y != label

X_new = X[mask]
y_new = y[mask]

# Save back
np.save(X_PATH, X_new)
np.save(Y_PATH, y_new)

print(f"Deleted label {label}")
print("New dataset size:", X_new.shape, y_new.shape)
