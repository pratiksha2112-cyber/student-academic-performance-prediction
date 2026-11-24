
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Academic Performance System",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"

st.title("🎓 Student Academic Performance Analysis and Prediction System")

st.write(
    "A Machine Learning based application "
    "to analyze and predict student academic performance."
)

# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH, sep=";")

    data = data.dropna()
    data = data.drop_duplicates()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    return data


try:
    model = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

X_raw = df.drop(columns=["G3"])
y = df["G3"]

categorical_cols = X_raw.select_dtypes(
    include=["object"]
).columns.tolist()

X = pd.get_dummies(
    X_raw,
    columns=categorical_cols,
    drop_first=True
)

feature_columns = list(model.feature_names_in_)

X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


# ==========================================
# 4. MODEL EVALUATION
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Student Prediction",
        "Model Evaluation"
    ]
)


# ==========================================
# 6. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("📊 Student Performance Dashboard")

    st.write(
        "Overview of the student dataset and academic performance."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric("Total Features", len(X.columns))
    col3.metric("Average Grade", round(df["G3"].mean(), 2))
    col4.metric("Maximum Grade", int(df["G3"].max()))

    st.divider()

    # Grade Distribution

    st.subheader("1. Final Grade Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["G3"], bins=15, edgecolor="black")

    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")

    st.pyplot(fig)
    plt.close(fig)

    # Study Time Analysis

    st.subheader("2. Study Time vs Final Grade")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(df["studytime"], df["G3"])

    ax.set_xlabel("Study Time")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("Study Time vs Academic Performance")

    st.pyplot(fig)
    plt.close(fig)

    # Absences Analysis

    st.subheader("3. Absences vs Final Grade")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(df["absences"], df["G3"])

    ax.set_xlabel("Absences")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("Absences vs Final Grade")

    st.pyplot(fig)
    plt.close(fig)

    # G2 vs G3

    st.subheader("4. Previous Grade vs Final Grade")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(df["G2"], df["G3"])

    ax.set_xlabel("Second Period Grade (G2)")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("G2 vs G3")

    st.pyplot(fig)
    plt.close(fig)

    # ======================================
    # CORRELATION HEATMAP WITH VALUES
    # ======================================

    st.subheader("5. Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90
    )

    ax.set_yticklabels(correlation.columns)

    # Display numerical correlation values

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):

            value = correlation.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation Value"
    )

    ax.set_title("Correlation Between Numerical Features")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Dataset Preview

    st.subheader("6. Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==========================================
# 7. STUDENT PREDICTION AND REPORT
# ==========================================

elif page == "Student Prediction":

    st.header("🧑‍🎓 Student Prediction")

    st.write(
        "Enter student information to predict the final grade."
    )

    with st.form("student_form"):

        st.subheader("Personal Information")

        age = st.number_input("Age", 15, 22, 17)

        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["LE3", "GT3"])
        Pstatus = st.selectbox("Parent Status", ["T", "A"])

        st.subheader("Family Information")

        Medu = st.selectbox(
            "Mother's Education",
            [0, 1, 2, 3, 4]
        )

        Fedu = st.selectbox(
            "Father's Education",
            [0, 1, 2, 3, 4]
        )

        Mjob = st.selectbox(
            "Mother's Job",
            ["at_home", "health", "services", "teacher", "other"]
        )

        Fjob = st.selectbox(
            "Father's Job",
            ["at_home", "health", "services", "teacher", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        st.subheader("Academic Information")

        traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = st.selectbox("Study Time", [1, 2, 3, 4])
        failures = st.selectbox("Past Failures", [0, 1, 2, 3, 4])

        G1 = st.number_input("First Period Grade (G1)", 0, 20, 12)
        G2 = st.number_input("Second Period Grade (G2)", 0, 20, 13)

        reason = st.selectbox(
            "Reason for Choosing School",
            ["course", "home", "reputation", "other"]
        )

        st.subheader("Lifestyle and Support")

        famrel = st.selectbox("Family Relationship", [1, 2, 3, 4, 5])
        freetime = st.selectbox("Free Time", [1, 2, 3, 4, 5])
        goout = st.selectbox("Going Out", [1, 2, 3, 4, 5])
        Dalc = st.selectbox("Workday Alcohol Consumption", [1, 2, 3, 4, 5])
        Walc = st.selectbox("Weekend Alcohol Consumption", [1, 2, 3, 4, 5])
        health = st.selectbox("Health", [1, 2, 3, 4, 5])

        absences = st.number_input("Absences", 0, 100, 2)

        schoolsup = st.selectbox("School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Paid Classes", ["yes", "no"])
        activities = st.selectbox("Extra Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

        submitted = st.form_submit_button(
            "Predict Student Performance"
        )

    if submitted:

        student_data = {
            "age": age,
            "Medu": Medu,
            "Fedu": Fedu,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
            "G1": G1,
            "G2": G2,
            "school": school,
            "sex": sex,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic
        }

        student_df = pd.DataFrame([student_data])

        student_df = pd.get_dummies(
            student_df,
            columns=categorical_cols,
            drop_first=True
        )

        student_df = student_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = float(model.predict(student_df)[0])

        prediction = float(
            np.clip(prediction, 0, 20)
        )

        # Performance Category

        if prediction >= 16:
            category = "Excellent"

        elif prediction >= 12:
            category = "Good"

        elif prediction >= 10:
            category = "Average"

        elif prediction >= 8:
            category = "Needs Improvement"

        else:
            category = "Poor"

        # Student Report

        st.divider()

        st.header("📄 Student Performance Report")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Final Grade",
            f"{prediction:.2f} / 20"
        )

        col2.metric(
            "Performance Category",
            category
        )

        st.subheader("Student Academic Details")

        report_details = pd.DataFrame({
            "Parameter": [
                "Age",
                "Study Time",
                "Past Failures",
                "Absences",
                "First Period Grade",
                "Second Period Grade"
            ],
            "Value": [
                age,
                studytime,
                failures,
                absences,
                G1,
                G2
            ]
        })

        st.table(report_details)

        # Suggestions

        st.subheader("Performance Analysis and Suggestions")

        suggestions = []

        if studytime <= 2:
            suggestions.append(
                "Consider increasing daily study time."
            )

        if absences > 10:
            suggestions.append(
                "Work on improving class attendance."
            )

        if failures > 0:
            suggestions.append(
                "Focus on subjects where difficulties occurred."
            )

        if G2 < 10:
            suggestions.append(
                "Focus on improving previous academic performance."
            )

        if not suggestions:
            suggestions.append(
                "The entered academic indicators look positive."
            )

        for suggestion in suggestions:
            st.write("•", suggestion)

        st.info(
            "This is an estimated grade based on the trained ML model. "
            "It is not a guaranteed result."
        )

        # Download Report

        report = pd.DataFrame([{
            "Age": age,
            "Study Time": studytime,
            "Absences": absences,
            "G1": G1,
            "G2": G2,
            "Predicted G3": round(prediction, 2),
            "Performance Category": category
        }])

        st.download_button(
            label="📥 Download Student Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="student_performance_report.csv",
            mime="text/csv"
        )


# ==========================================
# 8. MODEL EVALUATION PAGE
# ==========================================

elif page == "Model Evaluation":

    st.header("📈 Model Evaluation")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("MAE", round(mae, 4))
    col2.metric("MSE", round(mse, 4))
    col3.metric("RMSE", round(rmse, 4))
    col4.metric("R² Score", round(r2, 4))

    st.write(
        f"R² Score: The model explains approximately "
        f"{r2 * 100:.2f}% of the variation in final grades."
    )

    st.divider()

    st.subheader("Actual vs Predicted Grades")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_test, test_predictions)

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    ax.set_xlabel("Actual Grades")
    ax.set_ylabel("Predicted Grades")
    ax.set_title("Actual vs Predicted Grades")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Evaluation Summary")

    st.write("Mean Absolute Error:", round(mae, 4))
    st.write("Mean Squared Error:", round(mse, 4))
    st.write("Root Mean Squared Error:", round(rmse, 4))
    st.write("R² Score:", round(r2, 4))

    st.caption(
        "R² is a measure of explained variance, not classification accuracy."
    )
