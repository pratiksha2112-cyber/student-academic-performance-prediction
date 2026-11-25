import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import streamlit as st

import joblib


# Load student dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Display first five rows
print(df.head())

# Check dataset dimensions
print("Dataset Shape:", df.shape)

# Check column information
print(df.info())
#EDA(exploratory data analysis)

# Display first 5 rows
print(df.head())

# Display number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Display dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

#DATA CLEANING



# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
df = df.dropna()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip()

print(df.describe())

#data visualization
sns.set_style("whitegrid")

# 1. FINAL GRADE DISTRIBUTION
#kitne students k final marks G3 hai

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="G3",
    bins=20,
    kde=True,
    color="skyblue"
)

plt.title("Distribution of Student Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()
#comparison of g1 g2 g3
# 2. COMPARISON OF STUDENT GRADES

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df[["G1", "G2", "G3"]]
)

plt.title("Comparison of G1, G2 and G3 Grades")
plt.xlabel("Grade Categories")
plt.ylabel("Marks")

plt.show()

# 3. STUDY TIME VS FINAL GRADE

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="studytime",
    y="G3"
)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.show()

# 4. ABSENCES VS FINAL GRADE

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="absences",
    y="G3"
)

plt.title("Student Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.show()

# 5. CORRELATION HEATMAP

plt.figure(figsize=(14, 10))

correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Student Performance")

plt.tight_layout()
plt.show()


# FEATURE SELECTION

X = df.drop(columns=["G3"])
y = df["G3"]

# Display selected features
print("\nSelected Features:")
print(X.columns)

# Display target variable
print("\nTarget Variable:")
print(y.name)

# Display shape of X and y
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# DATA PREPROCESSING

# Identify numerical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:")
print(list(numerical_cols))

print("\nCategorical Columns:")
print(list(categorical_cols))

print("\nNumber of Numerical Columns:", len(numerical_cols))
print("Number of Categorical Columns:", len(categorical_cols))


# CATEGORICAL ENCODING

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Display first five rows
print("\nData After Encoding:")
print(X.head())

# Display new dataset shape
print("\nNew Dataset Shape:", X.shape)

# Check remaining categorical columns
print("\nRemaining Categorical Columns:")
print(X.select_dtypes(include=["object"]).columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display shapes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

model = LinearRegression()
print("\nLinear Regression Model Created Successfully!")

# TRAIN THE MODEL

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# MAKE PREDICTIONS

y_pred = model.predict(X_test)

print("\nPredicted Grades:")
print(y_pred[:10])

print("\nActual Grades:")
print(y_test[:10].values)
#evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)

#desision tree regressor model
#  Create Model
dt_model = DecisionTreeRegressor(random_state=42)

#  Train Model
dt_model.fit(X_train, y_train)

print("Decision Tree Model Trained Successfully!")

#  Make Predictions
dt_pred = dt_model.predict(X_test)

print("\nPredicted Grades:")
print(dt_pred[:10])

print("\nActual Grades:")
print(y_test[:10].values)
#evaluation
dt_mae = mean_absolute_error(y_test, dt_pred)

dt_mse = mean_squared_error(y_test, dt_pred)

dt_rmse = np.sqrt(dt_mse)

dt_r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree Evaluation:")
print("MAE:", dt_mae)
print("MSE:", dt_mse)
print("RMSE:", dt_rmse)
print("R-squared:", dt_r2)

# Compare both models

comparison = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],

    "Linear Regression": [
        mae,
        mse,
        rmse,
        r2
    ],

    "Decision Tree Regressor": [
        dt_mae,
        dt_mse,
        dt_rmse,
        dt_r2
    ]
})

print("\nModel Comparison:")
print(comparison)

# CROSS VALIDATION

# Linear Regression Cross Validation
lr_cv_scores = cross_val_score(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

# Decision Tree Cross Validation
dt_cv_scores = cross_val_score(
    DecisionTreeRegressor(random_state=42),
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\nLinear Regression CV Scores:")
print(lr_cv_scores)
print("Average LR CV Score:", lr_cv_scores.mean())

print("\nDecision Tree CV Scores:")
print(dt_cv_scores)
print("Average DT CV Score:", dt_cv_scores.mean())


# LINEAR REGRESSION OVERFITTING CHECK

lr_train_pred = model.predict(X_train)
lr_test_pred = model.predict(X_test)

print("\nLinear Regression Overfitting Check:")
print("Training R2:", r2_score(y_train, lr_train_pred))
print("Testing R2:", r2_score(y_test, lr_test_pred))
# DECISION TREE OVERFITTING CHECK

dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

print("\nDecision Tree Overfitting Check:")
print("Training R2:", r2_score(y_train, dt_train_pred))
print("Testing R2:", r2_score(y_test, dt_test_pred))

# FINAL MODEL SELECTION

final_model = model  # Linear Regression

print("\nFinal Selected Model: Linear Regression")

final_predictions = final_model.predict(X_test)

print("\nFinal Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, final_predictions))
print("MSE:", mean_squared_error(y_test, final_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, final_predictions)))
print("R2 Score:", r2_score(y_test, final_predictions))
# SAVE FINAL MODEL

joblib.dump(final_model, "student_performance_model.pkl")

print("\nFinal Model Saved Successfully!")



# LOAD SAVED MODEL

loaded_model = joblib.load("student_performance_model.pkl")

print("Model Loaded Successfully!")

# NEW STUDENT DATA

new_student = X.iloc[[0]].copy()

prediction = loaded_model.predict(new_student)

print("\nPredicted Final Grade (G3):", prediction[0])

print(X.columns.tolist())
# MANUAL NEW STUDENT PREDICTION

new_student = {
    'age': 17,
    'Medu': 3,
    'Fedu': 2,
    'traveltime': 1,
    'studytime': 2,
    'failures': 0,
    'famrel': 4,
    'freetime': 3,
    'goout': 2,
    'Dalc': 1,
    'Walc': 2,
    'health': 4,
    'absences': 2,
    'G1': 12,
    'G2': 13,
    'school_MS': 0,
    'sex_M': 0,
    'address_U': 1,
    'famsize_LE3': 0,
    'Pstatus_T': 1,
    'Mjob_health': 0,
    'Mjob_other': 0,
    'Mjob_services': 1,
    'Mjob_teacher': 0,
    'Fjob_health': 0,
    'Fjob_other': 1,
    'Fjob_services': 0,
    'Fjob_teacher': 0,
    'reason_home': 0,
    'reason_other': 0,
    'reason_reputation': 1,
    'guardian_mother': 1,
    'guardian_other': 0,
    'schoolsup_yes': 0,
    'famsup_yes': 1,
    'paid_yes': 0,
    'activities_yes': 1,
    'nursery_yes': 1,
    'higher_yes': 1,
    'internet_yes': 1,
    'romantic_yes': 0
}

new_student_df = pd.DataFrame([new_student])

predicted_grade = loaded_model.predict(new_student_df)

print("\nPredicted Final Grade (G3):", predicted_grade[0])



# USER INPUT PREDICTION

print("\n===== STUDENT ACADEMIC PERFORMANCE PREDICTION =====")
age = st.number_input(
    "Enter Age:",
    min_value=15,
    max_value=25,
    value=18,
    step=1
)
# Numerical Inputs (Streamlit widgets)
Medu = st.number_input("Mother's Education (0-4):", min_value=0, max_value=4, value=2, step=1)
Fedu = st.number_input("Father's Education (0-4):", min_value=0, max_value=4, value=2, step=1)
traveltime = st.number_input("Travel Time (1-4):", min_value=1, max_value=4, value=1, step=1)
studytime = st.number_input("Study Time (1-4):", min_value=1, max_value=4, value=2, step=1)
failures = st.number_input("Past Failures (0-4):", min_value=0, max_value=4, value=0, step=1)
famrel = st.number_input("Family Relationship (1-5):", min_value=1, max_value=5, value=4, step=1)
freetime = st.number_input("Free Time (1-5):", min_value=1, max_value=5, value=3, step=1)
goout = st.number_input("Going Out (1-5):", min_value=1, max_value=5, value=2, step=1)
Dalc = st.number_input("Workday Alcohol Consumption (1-5):", min_value=1, max_value=5, value=1, step=1)
Walc = st.number_input("Weekend Alcohol Consumption (1-5):", min_value=1, max_value=5, value=1, step=1)
health = st.number_input("Health Status (1-5):", min_value=1, max_value=5, value=3, step=1)
absences = st.number_input("Number of Absences:", min_value=0, max_value=100, value=0, step=1)
G1 = st.number_input("First Period Grade (0-20):", min_value=0, max_value=20, value=10, step=1)
G2 = st.number_input("Second Period Grade (0-20):", min_value=0, max_value=20, value=10, step=1)

# Categorical Inputs (Streamlit dropdowns)
school = st.selectbox("School:", ["GP", "MS"])
sex = st.selectbox("Sex:", ["F", "M"])
address = st.selectbox("Address:", ["U", "R"])
famsize = st.selectbox("Family Size:", ["LE3", "GT3"])
Pstatus = st.selectbox("Parent Status:", ["T", "A"])

Mjob = st.selectbox("Mother's Job:", ["teacher", "health", "services", "other", "at_home"])
Fjob = st.selectbox("Father's Job:", ["teacher", "health", "services", "other", "at_home"])
reason = st.selectbox("Reason for Choosing School:", ["home", "reputation", "course", "other"])
guardian = st.selectbox("Guardian:", ["mother", "father", "other"])

schoolsup = st.selectbox("Extra Educational Support:", ["yes", "no"])
famsup = st.selectbox("Family Educational Support:", ["yes", "no"])
paid = st.selectbox("Extra Paid Classes:", ["yes", "no"])
activities = st.selectbox("Extra Activities:", ["yes", "no"])
nursery = st.selectbox("Attended Nursery School:", ["yes", "no"])
higher = st.selectbox("Wants Higher Education:", ["yes", "no"])
internet = st.selectbox("Internet Access:", ["yes", "no"])
romantic = st.selectbox("In a Romantic Relationship:", ["yes", "no"])




# Create input dictionary
user_data = {
    'age': age, 'Medu': Medu, 'Fedu': Fedu,
    'traveltime': traveltime, 'studytime': studytime,
    'failures': failures, 'famrel': famrel,
    'freetime': freetime, 'goout': goout,
    'Dalc': Dalc, 'Walc': Walc, 'health': health,
    'absences': absences, 'G1': G1, 'G2': G2,

    'school_MS': int(school == 'MS'),
    'sex_M': int(sex == 'M'),
    'address_U': int(address == 'U'),
    'famsize_LE3': int(famsize == 'LE3'),
    'Pstatus_T': int(Pstatus == 'T'),

    'Mjob_health': int(Mjob == 'health'),
    'Mjob_other': int(Mjob == 'other'),
    'Mjob_services': int(Mjob == 'services'),
    'Mjob_teacher': int(Mjob == 'teacher'),

    'Fjob_health': int(Fjob == 'health'),
    'Fjob_other': int(Fjob == 'other'),
    'Fjob_services': int(Fjob == 'services'),
    'Fjob_teacher': int(Fjob == 'teacher'),

    'reason_home': int(reason == 'home'),
    'reason_other': int(reason == 'other'),
    'reason_reputation': int(reason == 'reputation'),

    'guardian_mother': int(guardian == 'mother'),
    'guardian_other': int(guardian == 'other'),

    'schoolsup_yes': int(schoolsup == 'yes'),
    'famsup_yes': int(famsup == 'yes'),
    'paid_yes': int(paid == 'yes'),
    'activities_yes': int(activities == 'yes'),
    'nursery_yes': int(nursery == 'yes'),
    'higher_yes': int(higher == 'yes'),
    'internet_yes': int(internet == 'yes'),
    'romantic_yes': int(romantic == 'yes')
}

# Convert into DataFrame
user_df = pd.DataFrame([user_data])

# Arrange columns in training order
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict Grade
predicted_grade = loaded_model.predict(user_df)

print("\n===== PREDICTION RESULT =====")
print("Predicted Final Grade (G3):", round(predicted_grade[0], 2), "/ 20")


print("\n===== FINAL MODEL EVALUATION =====")
print("MAE:", mean_absolute_error(y_test, final_predictions))
print("MSE:", mean_squared_error(y_test, final_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, final_predictions)))
print("R2 Score:", r2_score(y_test, final_predictions))

grade = predicted_grade[0]

if grade >= 16:
    category = "Excellent"
elif grade >= 12:
    category = "Good"
elif grade >= 10:
    category = "Average"
elif grade >= 8:
    category = "Needs Improvement"
else:
    category = "Poor"

print("\n===== STUDENT PERFORMANCE REPORT =====")
print("Predicted Final Grade:", round(grade, 2), "/ 20")
print("Performance Category:", category)

print("\n===== STUDENT PERFORMANCE ANALYSIS =====")

if studytime <= 2:
    print("Suggestion: Increase daily study time.")

if absences > 10:
    print("Suggestion: Improve class attendance.")

if failures > 0:
    print("Suggestion: Focus on previously difficult subjects.")

if G2 < 10:
    print("Suggestion: Improve performance in previous term.")

if G1 >= 10 and G2 >= 10:
    print("Observation: Student has maintained passing grades.")

if studytime > 2 and absences <= 10 and failures == 0:
    print("Observation: Student shows positive academic indicators.")
