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
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# 1. Create models directory
# ==============================

if not os.path.exists("models"):
    os.makedirs("models")

# ==============================
# 2. Load Dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv("data/cardio_train.csv", sep=';')

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

# ==============================
# 7. Random Forest Model
# ==============================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_acc)

joblib.dump(rf, "models/rf_model.pkl")

# ==============================
# 8. XGBoost Model
# ==============================

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train_scaled, y_train)

y_pred_xgb = xgb.predict(X_test_scaled)

xgb_acc = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_acc)

joblib.dump(xgb, "models/xgb_model.pkl")

# ==============================
# 9. Deep Learning Model (ANN)
# ==============================

print("\nTraining ANN...")

ann = Sequential()
ann.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
ann.add(Dropout(0.3))
ann.add(Dense(64, activation='relu'))
ann.add(Dropout(0.2))
ann.add(Dense(32, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = ann.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

loss, ann_acc = ann.evaluate(X_test_scaled, y_test)
print("ANN Accuracy:", ann_acc)

ann.save("models/ann_model.h5")

# ==============================
# 10. Model Comparison
# ==============================

print("\nModel Comparison:")
print(f"Random Forest Accuracy : {rf_acc:.4f}")
print(f"XGBoost Accuracy       : {xgb_acc:.4f}")
print(f"ANN Accuracy           : {ann_acc:.4f}")

# ==============================
# 11. Confusion Matrix (Best Model)
# ==============================

best_model_name = max(
    [("RF", rf_acc), ("XGB", xgb_acc), ("ANN", ann_acc)],
    key=lambda x: x[1]
)[0]

print("\nBest Model:", best_model_name)

if best_model_name == "RF":
    best_model = rf
    y_pred = y_pred_rf
elif best_model_name == "XGB":
    best_model = xgb
    y_pred = y_pred_xgb
else:
    y_pred = (ann.predict(X_test_scaled) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix.png")
plt.close()

# ==============================
# 12. ROC Curve
# ==============================

if best_model_name == "ANN":
    y_probs = ann.predict(X_test_scaled)
else:
    y_probs = best_model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - " + best_model_name)
plt.legend()
plt.savefig("models/roc_curve.png")
plt.close()

print("AUC Score:", auc_score)

# ==============================
# 13. Save Best Model
# ==============================

if best_model_name != "ANN":
    joblib.dump(best_model, "models/best_model.pkl")

print("\nTraining Completed Successfully!")