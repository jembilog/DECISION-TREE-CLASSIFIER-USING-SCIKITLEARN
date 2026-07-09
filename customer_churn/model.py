import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("customer_churn.csv")

#missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
df["InternetService"] = df["InternetService"].fillna(df["InternetService"].mode()[0])
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())

#standardize
df["Gender"] = df["Gender"].str.strip()
df["Gender"] = df["Gender"].str.title()

#duplicates
df = df.drop_duplicates()

#categorical to num
gender_encoder = LabelEncoder()
contract_encoder = LabelEncoder()
internet_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Contract"] = contract_encoder.fit_transform(df["Contract"])
df["InternetService"] = internet_encoder.fit_transform(df["InternetService"])
df["Churn"] = target_encoder.fit_transform(df["Churn"])


X = df[["Age", "Gender", "MonthlyCharges", "Contract", "InternetService", "Tenure"]]
y = df["Churn"]

X_train, X_test, y_train , y_test = train_test_split(
    X, y , test_size=0.2, random_state= 42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100
print("Accuracy:",accuracy,"%")

new_customer =pd.DataFrame({
    "Age" : [35],
    "Gender" : [0],
    "MonthlyCharges" : [80],
    "Contract": [1],
    "InternetService" : [0],
    "Tenure": [18]
})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Stay")
else:
    print("Leave")

importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(feature, score)






















#gender_encoder
# contract_encoder
# internet_encoder
# target_encoder
