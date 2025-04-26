import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_excel('emails.xlsx')

le = LabelEncoder()
df['sender_address'] = le.fit_transform(df['sender_address'])

X = df.drop('spam', axis=1)
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Deploy (predict) for a new incoming email
new_email = pd.DataFrame([[10, 300, 2, le.transform(['unknown_sender'])[0]]], columns=X.columns)
prediction = model.predict(new_email)
print(f"Predicted (0 = not spam, 1 = spam): {prediction[0]}")
