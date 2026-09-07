import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. CREATE DATASET
# ==========================================

data = {
    "Study_Hours": [
        2, 5, 1, 6, 3, 7, 2, 4, 8, 1,
        5, 3, 6, 2, 7, 4, 1, 5, 6, 3
    ],

    "Attendance": [
        60, 85, 50, 90, 70, 95, 65, 80, 98, 45,
        88, 72, 92, 55, 94, 78, 40, 86, 91, 68
    ],

    "Previous_Marks": [
        45, 75, 35, 80, 55, 90, 48, 70, 95, 30,
        82, 60, 85, 42, 88, 68, 25, 78, 84, 52
    ],

    "Assignment_Score": [
        50, 80, 40, 85, 60, 92, 45, 75, 96, 35,
        85, 65, 88, 48, 90, 72, 30, 82, 86, 58
    ],

    "Result": [
        "Fail", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Fail",
        "Pass", "Pass", "Pass", "Fail", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass"
    ]
}


# Convert data into DataFrame
df = pd.DataFrame(data)


print("==========================================")
print("      STUDENT PASS/FAIL PREDICTION")
print("==========================================")


# ==========================================
# 2. DISPLAY DATASET
# ==========================================

print("\nStudent Dataset:")
print(df.to_string(index=False))


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Score"
    ]
]

y = df["Result"]


# ==========================================
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 5. CREATE AND TRAIN MODEL
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)


# ==========================================
# 6. PREDICT TEST DATA
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# ==========================================
# 9. OPTIONAL NEW STUDENT PREDICTION
# ==========================================

while True:

    print("\n==========================================")
    print("          NEW STUDENT PREDICTION")
    print("==========================================")

    choice = input(
        "Do you want to add/predict a student? (yes/no): "
    )

    if choice.lower() == "no":

        print("\nNo new student added.")
        print("Program completed.")
        break


    elif choice.lower() == "yes":

        # Take student details
        study_hours = float(
            input("Enter Study Hours: ")
        )

        attendance = float(
            input("Enter Attendance (%): ")
        )

        previous_marks = float(
            input("Enter Previous Marks: ")
        )

        assignment_score = float(
            input("Enter Assignment Score: ")
        )


        # Create DataFrame for new student
        new_student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks],
            "Assignment_Score": [assignment_score]
        })


        # Predict result
        prediction = model.predict(new_student)


        # Display prediction
        print("\n------------------------------------------")
        print("Predicted Result:", prediction[0])
        print("------------------------------------------")


    else:

        print("\nPlease enter only 'yes' or 'no'.")