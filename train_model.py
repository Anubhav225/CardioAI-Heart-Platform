import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "heart_disease_data.csv"
MODEL_PATH = "heart_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Use target column if available, otherwise use the last column
target_col = "target" if "target" in df.columns else df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {MODEL_PATH}")