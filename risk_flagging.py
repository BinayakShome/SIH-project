import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load CSVs ----------
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
    else:
        return "High Risk"

df["risk_rule"] = df.apply(rule_based_risk, axis=1)

# ---------- ML Risk Prediction ----------
X = df[["attendance_percent","avg_marks","failed_attempts","fee_paid_percent"]]
y = df["risk_rule"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
df["risk_ml"] = le.inverse_transform(clf.predict(X))

print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------- Save final CSV ----------
df.to_csv(r"c:\Users\KIIT0001\Documents\GitHub\SIH-project\data\final_with_risk.csv", index=False)

# ---------- VISUALIZATIONS ----------

# 1️⃣ Bar chart: Number of students per risk level
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="risk_rule", order=["Low Risk","Medium Risk","High Risk"], palette="Set2")
plt.title("Number of Students per Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Number of Students")
plt.show()

# 2️⃣ Histogram: Attendance distribution colored by risk
plt.figure(figsize=(8,4))
sns.histplot(data=df, x="attendance_percent", hue="risk_rule", multiple="stack", palette="Set1", bins=20)
plt.title("Attendance Distribution by Risk Level")
plt.xlabel("Attendance Percent")
plt.ylabel("Count")
plt.show()

# 3️⃣ Heatmap: Correlation matrix
plt.figure(figsize=(6,5))
corr = df[["attendance_percent","avg_marks","failed_attempts","fee_paid_percent"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# 4️⃣ Stacked bar: Failed attempts vs risk
plt.figure(figsize=(8,4))
failed_risk = df.groupby(["failed_attempts","risk_rule"]).size().unstack(fill_value=0)
failed_risk[["Low Risk","Medium Risk","High Risk"]].plot(kind="bar", stacked=True, colormap="Set3")
plt.title("Failed Attempts vs Risk Level")
plt.xlabel("Number of Failed Attempts")
plt.ylabel("Number of Students")
plt.show()


