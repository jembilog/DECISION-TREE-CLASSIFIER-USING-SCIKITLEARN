import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("spam_email.csv")

df["Email"] = df["Email"].fillna("")
df["Email"] = df["Email"].str.strip()
df["Email"] = df["Email"].str.lower()
df = df.drop_duplicates()


#convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Email"])

#encode
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Spam"])

print(label_encoder.classes_)

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

print("\n========== RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
new_email = ["Congratulations! You won a free iPhone. Click now!"]
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("\n========== NEW EMAIL ==========")

if prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")
