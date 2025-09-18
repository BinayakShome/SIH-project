import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_merge_data, generate_labels

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

print("🔄 Loading and preprocessing data...")
df = load_and_merge_data()
df = generate_labels(df)
print(f"✅ Data loaded. Shape: {df.shape}")

# Encode categorical
df["payment_status"] = LabelEncoder().fit_transform(df["payment_status"].astype(str))

# Features & target
X = df.drop(columns=["student_id", "name", "dob", "program", "dropout"], errors="ignore")
y = df["dropout"]

print("🔄 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🔄 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "models/dropout_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("💾 Model and scaler saved to models/")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Training complete. Accuracy: {accuracy:.4f}")
print(report)

# Save report to file
with open("models/training_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("📄 Training report saved at models/training_report.txt")

