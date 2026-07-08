import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("heart_disease.csv")

X = df[["Age", "Cholesterol", "BloodPressure", "HeartRate"]]

#target
y = df["Disease"]

X_train , X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    random_state=42
)

#train
model.fit(X_train, y_train)

#predict
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

new_patient = [[50,238,140,84]]

prediction = model.predict(new_patient)
print("\nPrediction:")
if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("Healthy")


#if you want to visualize Tree
# plt.figure(figsize=(15,8))

# plot_tree(
#     model,
#     feature_names=X.columns,
#     class_names=["Healthy", "Disease"],
#     filled=True,
#     rounded=True
# )

# plt.show()
