import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
df = pd.read_csv("student_pass.csv")

# Features
X = df[["StudyHours", "Attendance"]]

# Target
y = df["Pass"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = DecisionTreeClassifier()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))

# New Prediction
new_student = [[5,82]]

result = model.predict(new_student)

print("\nPrediction:")

if result[0] == 1:
    print("Pass")
else:
    print("Fail")

plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names = ["StudyHours", "Attendance"],
    class_names = ["Fail", "Pass"],
    filled=True
)
plt.show()