import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model = DecisionTreeClassifier(random_state=42)
model = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42
)

#train
model.fit(X_train, y_train)

#predict
predictions = model.predict(X_test)

#accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))

#Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nPrediction:", iris.target_names[prediction[0]])

#Visualize the Decision Tree
plt.figure(figsize=(16, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.show()