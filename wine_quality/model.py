import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("wine_quality.csv")

X = df[["Acidity", "Sugar", "pH", "Alcohol"]]
y = df["Quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,predictions))
print("\nClassification Report")
print(classification_report(y_test,predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictions))

new_wine = [[7.3, 1.8,3.35,11.4]]
prediction = model.predict(new_wine)

print("\nPredicted Wine Quality:", prediction[0])

plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(i) for i in sorted(df["Quality"].unique())],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.show()
