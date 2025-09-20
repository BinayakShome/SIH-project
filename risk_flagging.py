import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------- Load CSVs with absolute paths ----------
students = pd.read_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\students (2).csv")
attendance = pd.read_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\attendance (1).csv")
marks = pd.read_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\marks.csv")
fees = pd.read_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\fees (1).csv")

# ---------- Merge datasets ----------
df = students.merge(attendance, on="student_id") \
             .merge(marks, on="student_id") \
             .merge(fees, on="student_id")

# ---------- Rule-Based Risk Flagging ----------
def rule_based_risk(row):
    score = 0
    if row["attendance_percent"] < 75:
        score += 1
    if row["avg_marks"] < 50 or row["failed_attempts"] > 1:
        score += 1
    if row["fee_paid_percent"] < 70:
        score += 1

    if score <= 1:
        return "Low Risk"
    elif score == 2:
        return "Medium Risk"
    else:  # score == 3
        return "High Risk"

df["risk_rule"] = df.apply(rule_based_risk, axis=1)

print("===== SAMPLE Rule-Based Risk Flagging =====")
print(df[['student_id','name','attendance_percent','avg_marks','failed_attempts','fee_paid_percent','risk_rule']].head(10))

# ---------- Machine Learning Risk Prediction ----------
# Features
X = df[["attendance_percent","avg_marks","failed_attempts","fee_paid_percent"]]
y = df["risk_rule"]  # ML target with 3 classes: Low/Medium/High Risk

# Encode string labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Low=0, Medium=1, High=2 (example mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n===== ML Classification Report =====")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Add ML predictions to dataframe
df["risk_ml"] = le.inverse_transform(clf.predict(X))

print("\n===== SAMPLE ML-Based Predictions =====")
print(df[['student_id','name','attendance_percent','avg_marks','failed_attempts','fee_paid_percent','risk_ml']].head(10))

# ---------- Save final dataset ----------
df.to_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\final_with_risk.csv", index=False)
print("\n✅ Merged dataset with risk flags saved as 'final_with_risk.csv'")

