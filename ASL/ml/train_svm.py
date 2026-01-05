import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Loaded data:", X.shape, y.shape)
print("Unique labels:", set(y.tolist()))

if len(set(y.tolist())) < 2:
    raise ValueError("Need at least 2 different letters to train the model")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Training accuracy:", acc)

# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved at:", MODEL_PATH)
print("Model file size:", os.path.getsize(MODEL_PATH), "bytes")
