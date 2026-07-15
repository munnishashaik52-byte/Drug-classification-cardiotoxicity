# ==============================
# AI-Based Drug Classification System
# train_model.py
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 1. Create models directory
# ==============================

if not os.path.exists("models"):
    os.makedirs("models")

# ==============================
# 2. Load Dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv("dataset/cardio_train.csv", sep=';')

print("\nDataset Shape:", df.shape)
print(df.head())

# ==============================
# 3. Data Preprocessing
# ==============================

print("\nPreprocessing data...")

# Drop ID column
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Convert age from days to years
df["age"] = df["age"] / 365

# Remove unrealistic blood pressure values
df = df[(df["ap_hi"] < 250) & (df["ap_lo"] < 200)]

# ==============================
# 4. Define Features & Target
# ==============================

X = df.drop("cardio", axis=1)
y = df["cardio"]

# ==============================
# 5. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. Feature Scaling
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print("\n[✓] Scaler saved to models/scaler.pkl")

# ==============================
# 7. Random Forest Model
# ==============================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_acc)

# ==============================
# 8. Save Best Model
# ==============================

joblib.dump(rf, "models/best_model.pkl")
print("\n[✓] Model saved to models/best_model.pkl")

# ==============================
# 9. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix.png")
plt.close()
print("[✓] Confusion matrix saved to models/confusion_matrix.png")

# ==============================
# 10. ROC Curve
# ==============================

y_probs = rf.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("models/roc_curve.png")
plt.close()
print("[✓] ROC curve saved to models/roc_curve.png")

# ==============================
# 11. Summary
# ==============================

print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Model Accuracy: {rf_acc:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print("\nFiles saved:")
print("  - models/scaler.pkl")
print("  - models/best_model.pkl")
print("  - models/confusion_matrix.png")
print("  - models/roc_curve.png")

