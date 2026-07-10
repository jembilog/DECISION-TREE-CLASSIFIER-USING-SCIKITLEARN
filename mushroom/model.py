import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("mushroom.csv")

columns_to_clean = ["CapShape", "CapColor", "Odor", "GillSize", "Habitat", "Edible"]
encoders = {}

for col in columns_to_clean:
    df[col] = df[col].str.strip().str.title()
    df[col] = df[col].fillna(df[col].mode()[0])

for column in df.columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

df = df.drop_duplicates()

for column, encoder in encoders.items():
    print(f"\n{column}")
    for original, encoded in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"  {original} -> {encoded}")


X = df[["CapShape", "CapColor", "Odor", "GillSize", "Habitat"]]
y = df["Edible"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    )
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


new_mushroom = pd.DataFrame({
    "CapShape": [0],
    "CapColor": [2],
    "Odor": [1],
    "GillSize": [1],
    "Habitat": [1]
})

prediction = model.predict(new_mushroom)
print(f"\nPrediction for new mushroom: {'Edible' if prediction[0] == 1 else 'Not edible'}")

plt.figure(figsize=(12, 8))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Not Edible', 'Edible'],
          filled=True,
          rounded=True)
plt.title("Mushroom Edibility Decision Tree")
plt.savefig("decision_tree.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nEncoded data preview:")
print(df.head(10))
