import json
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load training data
with open("training_data.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Preprocess data (same as train_model.ipynb)
def preprocess_data(df):
    df_processed = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    return np.array(df_processed)

X = preprocess_data(df)

# Load trained model
model = joblib.load("anomaly_model.joblib")

# Predict anomalies
preds = model.predict(X)  # 1: normal, -1: anomaly
scores = model.decision_function(X)  # Lower = more anomalous

# Print confidence scores for first 10 samples
print("Sample | Prediction | Confidence Score")
for i in range(10):
    status = "Normal" if preds[i] == 1 else "Anomaly"
    print(f"{i+1:6} | {status:8} | {scores[i]:.4f}")

# Apply PCA for 2D representation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot with color gradient for confidence
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scores, cmap='coolwarm', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Normal vs Anomaly Points (PCA 2D, Confidence Color)')
plt.colorbar(scatter, label='Confidence Score (decision_function)')
plt.tight_layout()
plt.show()
