# ==========================================
# STUDENT RESULT PREDICTION - ENHANCED
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv("student_data.csv")
print("\nDataset Loaded Successfully!")
print("\nFirst 5 Records:\n")
print(data.head())
# ==========================================
# PREPROCESSING
# ==========================================
encoder = LabelEncoder()
data['Result'] = encoder.fit_transform(data['Result'])
X = data[['Study_Hours', 'Attendance', 'Previous_Marks']]
y = data['Result']
# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
# ==========================================
# TRAIN MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")
# ==========================================
# EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# ==========================================
# CROSS VALIDATION
# ==========================================
scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Accuracy:")
print(scores)
print("\nAverage Accuracy: {:.2f}%".format(scores.mean() * 100))
# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
# ==========================================
# FEATURE IMPORTANCE
# ==========================================
importance = model.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(
    X.columns,
    importance
)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(model, "student_result_model.pkl")
print("\nModel Saved Successfully!")
# ==========================================
# USER PREDICTION
# ==========================================
print("\n===== STUDENT RESULT PREDICTION =====")
study = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))
marks = float(input("Enter Previous Marks: "))
new_student = [[study, attendance, marks]]
prediction = model.predict(new_student)
result = encoder.inverse_transform(prediction)
print("\nPredicted Result:", result[0])
# Probability
prob = model.predict_proba(new_student)
print("\nPrediction Probability:")
print("Fail Probability : {:.2f}%".format(prob[0][0]*100))
print("Pass Probability : {:.2f}%".format(prob[0][1]*100))