import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("titanic_dataset.csv")
# print(df.info())
# print(df.isnull().sum())
# print(df.duplicated().sum())

df["Sex"] = df["Sex"].str.strip()
df["Sex"] = df["Sex"].str.title()
df["Embarked"] = df["Embarked"].str.strip()
df["Embarked"] = df["Embarked"].str.title()

df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

le_embarked = LabelEncoder()
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

X = df[["PassengerClass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))

new_person = pd.DataFrame({
    "PassengerClass" : [3],
    "Sex" : [1],
    "Age" : [22],
    "Fare" : [10.2],
    "Embarked" : [3]
})
# print(dict(zip(le_sex.classes_,le_sex.transform(le_sex.classes_))))
# print(dict(zip(le_embarked.classes_,le_embarked.transform(le_embarked.classes_))))
prediction = model.predict(new_person)

if prediction[0] == 1:
    print("Survived")
else:
    print("deads")
