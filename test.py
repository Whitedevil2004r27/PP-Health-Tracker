import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------
# Load artifacts
# -------------------------
model = load_model("lstm_multiclass_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------------
# Features
# -------------------------
numerical_cols = ["age", "sleep_quality_index", "brain_fog_level",
                  "physical_pain_score", "stress_level", "depression_phq9_score",
                  "fatigue_severity_scale_score", "pem_duration_hours",
                  "hours_of_sleep_per_night", "pem_present"]

categorical_cols = ["gender", "work_status", "social_activity_level",
                    "exercise_frequency", "meditation_or_mindfulness"]

columns = numerical_cols + categorical_cols

root = tk.Tk()
root.title("Diagnosis Prediction")

entries = {}

# -------------------------
# Prediction function
# -------------------------
def predict():
    try:
        input_vals = []
        # Collect values
        for col in columns:
            if col in categorical_cols:
                val = entries[col].get()
                if val == '':
                    messagebox.showerror("Input Error", f"Select value for '{col}'")
                    return
                le = label_encoders[col]
                input_vals.append(le.transform([val])[0])
            else:
                val = entries[col].get()
                if val == '':
                    messagebox.showerror("Input Error", f"Enter value for '{col}'")
                    return
                input_vals.append(float(val))

        # Scale
        input_arr = np.array(input_vals).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)

        # Reshape for LSTM
        input_lstm = input_scaled.reshape((1, input_scaled.shape[1], 1))

        # Predict
        pred_probs = model.predict(input_lstm)
        pred_class = np.argmax(pred_probs, axis=1)[0]
        pred_label = target_encoder.inverse_transform([pred_class])[0]

        messagebox.showinfo("Prediction Result", f"Predicted Diagnosis: {pred_label}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------------
# Create input fields
# -------------------------
row = 0
# Numerical inputs
for col in numerical_cols:
    tk.Label(root, text=col).grid(row=row, column=0, sticky='w', padx=10, pady=5)
    ent = tk.Entry(root)
    ent.grid(row=row, column=1, padx=10, pady=5)
    entries[col] = ent
    row += 1

# Categorical inputs (dropdown)
for col in categorical_cols:
    tk.Label(root, text=col).grid(row=row, column=0, sticky='w', padx=10, pady=5)
    values = label_encoders[col].classes_.tolist()
    combo = ttk.Combobox(root, values=values, state="readonly")
    combo.grid(row=row, column=1, padx=10, pady=5)
    entries[col] = combo
    row += 1

# Predict button
tk.Button(root, text="Predict", command=predict).grid(row=row, column=0, columnspan=2, pady=20)

root.mainloop()
