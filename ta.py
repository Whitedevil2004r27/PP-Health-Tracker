# -------------------------
# Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("dataset.csv")

# -------------------------
# Handle Missing Values
# -------------------------
# Separate categorical & numerical columns
categorical_cols = ["gender", "work_status", "social_activity_level", 
                    "exercise_frequency", "meditation_or_mindfulness"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["diagnosis"]]

# Impute numerical columns with mean
imputer_num = SimpleImputer(strategy="mean")
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
joblib.dump(imputer_num, "num_imputer.pkl")

# Impute categorical columns with most frequent
imputer_cat = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
joblib.dump(imputer_cat, "cat_imputer.pkl")

# -------------------------
# Separate features & target
# -------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# -------------------------
# Encode Categorical Columns
# -------------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
joblib.dump(label_encoders, "label_encoders.pkl")

# Encode target labels
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
joblib.dump(target_encoder, "target_encoder.pkl")

# -------------------------
# Scale Numerical Features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# -------------------------
# Handle Imbalance with SMOTE
# -------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Reshape for LSTM input
X_res_lstm = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res_lstm, y_res, test_size=0.2, random_state=42
)

# -------------------------
from tensorflow.keras.layers import Dropout

# -------------------------
# Define LSTM Model (for multiclass)
# -------------------------
num_classes = len(np.unique(y))  # detect number of classes automatically

input_lstm = Input(shape=(X_train.shape[1], 1))
lstm = LSTM(128, return_sequences=False)(input_lstm)  # more units
dense = Dense(128, activation='relu')(lstm)
dense = Dropout(0.3)(dense)  # dropout to prevent overfitting
dense = Dense(64, activation='relu')(dense)
dense = Dropout(0.2)(dense)
final_output = Dense(num_classes, activation='softmax')(dense)  # softmax for multiclass

model = Model(inputs=input_lstm, outputs=final_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# Train Model
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=15,  # more epochs
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save model
save_model(model, "lstm_multiclass_model.h5")

# -------------------------
# Evaluation
# -------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # pick class with highest probability

print("\nEvaluation Results")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score (macro) :", f1_score(y_test, y_pred, average='macro'))
print("F1 Score (weighted) :", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# -------------------------
# ROC Curve (One-vs-Rest for multiclass)
# -------------------------
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve

y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(fpr[i], tpr[i], label=f"Class {target_encoder.classes_[i]} AUC = {roc_auc[i]:.3f}")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM Multiclass")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Confusion Matrix Heatmap
# -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix - LSTM Multiclass")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
