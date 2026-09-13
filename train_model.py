import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("dataset.csv")
X = df.drop(columns=["disease"])
y = df["disease"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])

# The small demo dataset is intended for learning, not clinical use.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model.fit(X_train, y_train)

print("Test accuracy:", round(model.score(X_test, y_test) * 100, 2), "%")
joblib.dump(model, "medical_svm.joblib")
print("Saved medical_svm.joblib")
