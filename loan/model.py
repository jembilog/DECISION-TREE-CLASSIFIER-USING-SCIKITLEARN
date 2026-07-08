import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("loan_approval.csv")
print("\n===== INFO =====")
print(df.info())
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
print("\n===== DUPLICATES =====")
print(df.duplicated().sum())

df["Employment"] = df["Employment"].str.strip() # remove extra spaces
df["Employment"] = df["Employment"].str.title()

#fill missing values

#categorical
df["Income"] = df["Income"].fillna(df["Income"].median())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())

#numerical
df["Employment"] = df["Employment"].fillna(df["Employment"].mode()[0])

#remove duplicates
df = df.drop_duplicates()

#encode categorical data
le_employment = LabelEncoder()
df["Employment"] = le_employment.fit_transform(df["Employment"])
# Employment Encoding:
# Employed = 0
# Self-Employed = 1
# Unemployed = 2

le_target = LabelEncoder()
df["LoanApproved"] = le_target.fit_transform(df["LoanApproved"])

#after preporcssing
print("\n===== CLEANED DATA =====")
print(df)
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

X = df[["Age", "Income", "CreditScore", "Employment"]]
y  = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\n========== RESULTS ==========")
print("Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nEmployment Classes:")
print(dict(zip(le_employment.classes_,
               le_employment.transform(le_employment.classes_))))

new_applicant = pd.DataFrame({
    "Age": [32],
    "Income": [45000],
    "CreditScore": [710],
    "Employment": [0]   # Change according to printed encoding
})#pwedeng array basta naka .values ang X and y

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

# plt.figure(figsize=(16,8))

# plot_tree(
#     model,
#     feature_names=X.columns,
#     class_names=le_target.classes_,
#     filled=True,
#     rounded=True,
#     fontsize=10
# )

# plt.show()
