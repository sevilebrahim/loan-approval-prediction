import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("loan_prediction.csv")

print(df.head())
print(df.describe())


df["married"] = df["married"].map({
    "Yes": 1,
    "No": 0
})

df["loan_approved"] = df["loan_approved"].map({
    "Yes": 1,
    "No": 0
})


X = [
    "age",
    "income",
    "credit_score",
    "married"
]

y = df["loan_approved"]


X_train, X_test, y_train, y_test = train_test_split(
    df[X],
    y,
    test_size=0.2,
    random_state=42
)


model = DecisionTreeClassifier()

model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)


data = pd.DataFrame(
    [[35, 50, 720, 1]],
    columns=X
)

data_prediction = model.predict(data)


if data_prediction[0] == 1:
    print("Loan approved.")
else:
    print("Loan not approved.")

