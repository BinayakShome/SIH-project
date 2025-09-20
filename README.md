# AI-Based Student Dropout Prediction and Risk Flagging System

## Overview
This project is designed to **predict at-risk students** using a combination of **rule-based scoring** and **machine learning** techniques. By analyzing student attendance, marks, failed attempts, and fee payment data, the system helps educators and counselors **identify students who may require early intervention**.

---

## Features
1. **Rule-Based Risk Flagging**
   - Assigns students a risk level (`Low`, `Medium`, `High`) using simple rules based on:
     - Attendance percentage
     - Average marks
     - Number of failed attempts
     - Fee payment percentage

2. **Machine Learning Prediction**
   - Uses a **Random Forest Classifier** to predict student risk based on historical data.
   - Compares rule-based risk with ML predictions for enhanced accuracy.

3. **Visualizations**
   - **Bar Chart:** Number of students per risk level  
   - **Histogram:** Attendance distribution by risk  
   - **Heatmap:** Feature correlation  
   - **Stacked Bar:** Failed attempts vs risk level  

4. **CSV Output**
   - Saves a combined dataset including **rule-based** and **ML-predicted** risk levels.

---

## Dataset
The project uses four CSV files stored in the `data` folder:

| File | Description |
|------|-------------|
| `students (2).csv` | Basic student details (ID, name, etc.) |
| `attendance (1).csv` | Attendance percentage per student |
| `marks.csv` | Average marks and failed attempts |
| `fees (1).csv` | Fee payment percentage |

> Note: Ensure all files are present in the `data` folder inside the project directory.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/BinayakShome/SIH-project.git

2. Install required packages:

pip install pandas scikit-learn matplotlib seaborn

Usage

1. Run the Python script:

python risk_flagging.py

2. The script will:

Merge all datasets

Assign rule-based risk

Train a Random Forest model and predict risk

Save final results as final_with_risk.csv in the data folder

Generate visualizations
