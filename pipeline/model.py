import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Load and Preprocess Data
# ============================
df = pd.read_excel("mix.xlsx")

# Drop non-informative columns
df.drop(columns=["sheep number", "video_time", "real Time"], inplace=True, errors="ignore")

# Fix typo if it exists
df.rename(columns={"behaivour": "behaviour"}, inplace=True)

# Remove rows with missing labels
df = df[df["behaviour"].notna()].copy()

# Label encoding
le = LabelEncoder()
df["behaviour"] = le.fit_transform(df["behaviour"])
label_names = [str(label) for label in le.classes_]

# ============================
# Feature Engineering
# ============================
# Magnitude
for i in range(1, 31):
    df[f'mag_{i}'] = np.sqrt(df[f'x_{i}']**2 + df[f'y_{i}']**2 + df[f'z_{i}']**2)

# Delta Magnitude
for i in range(1, 30):
    df[f'delta_mag_{i}'] = abs(df[f'mag_{i+1}'] - df[f'mag_{i}'])

# Statistical Aggregates
mag_cols = [f'mag_{i}' for i in range(1, 31)]
delta_cols = [f'delta_mag_{i}' for i in range(1, 30)]
df["mag_mean"] = df[mag_cols].mean(axis=1)
df["mag_std"] = df[mag_cols].std(axis=1)
df["delta_mean"] = df[delta_cols].mean(axis=1)
df["delta_std"] = df[delta_cols].std(axis=1)

# ============================
# Prepare Data
# ============================
X_flat = df.drop(columns=["behaviour"]).values
y = df["behaviour"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# Apply SMOTE to balance classes
print("Original class distribution:", np.bincount(y))
min_class_count = np.bincount(y).min()
k_neighbors = min(5, min_class_count - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("After SMOTE:", np.bincount(y_resampled))

# ============================
# Reshape for BLSTM
# ============================
X_seq = []
for row in X_resampled:
    sample = []
    for i in range(1, 31):
        x = row[df.columns.get_loc(f"x_{i}")]
        y_val = row[df.columns.get_loc(f"y_{i}")]
        z = row[df.columns.get_loc(f"z_{i}")]
        mag = row[df.columns.get_loc(f"mag_{i}")]
        delta = row[df.columns.get_loc(f"delta_mag_{i}")] if i < 30 else 0
        sample.append([x, y_val, z, mag, delta])
    X_seq.append(sample)

X = np.array(X_seq)
y_cat = to_categorical(y_resampled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

# ============================
# Build Bidirectional LSTM Model
# ============================
model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=40, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# ============================
# Evaluate
# ============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 6))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title("BLSTM Confusion Matrix with SMOTE (in %)")
plt.xlabel("Predicted Behaviour")
plt.ylabel("Actual Behaviour")
plt.tight_layout()
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

# ============================
# Accuracy and Loss Plots
# ============================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================
# Save Model
# ============================
model.save("C:/Users/HP/My Drive/SheepData/sheep1/sheep_blstm_model.h5")
print("✅ Model saved as sheep_blstm_model.h5")