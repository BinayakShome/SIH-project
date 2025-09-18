import pandas as pd

DATA_PATH = "Dummy_csvs/"

def load_and_merge_data():
    students = pd.read_csv(DATA_PATH + "students.csv")
    fees = pd.read_csv(DATA_PATH + "fees.csv")
    attendance = pd.read_csv(DATA_PATH + "attendance.csv")
    results = pd.read_csv(DATA_PATH + "results.csv")

    # Attendance summary
    attendance_summary = (
        attendance.groupby("student_id")["present"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "attendance_rate", "sum": "total_present", "count": "total_classes"})
    )

    # Results summary
    results["normalized_score"] = results["score"] / results["max_score"]
    results_summary = (
        results.groupby("student_id")["normalized_score"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_score", "min": "min_score", "max": "max_score", "count": "num_tests"})
    )

    # Merge datasets
    merged = (
        students.merge(attendance_summary, on="student_id", how="left")
        .merge(results_summary, on="student_id", how="left")
        .merge(fees[["student_id", "dues", "payment_status"]], on="student_id", how="left")
    )

    return merged

def generate_labels(df):
    def label(row):
        if row["attendance_rate"] < 0.6: return 1
        if row["avg_score"] < 0.4: return 1
        if row["dues"] > 500 or str(row["payment_status"]).lower() in ["late", "pending"]: return 1
        return 0
    df["dropout"] = df.apply(label, axis=1)
    return df
