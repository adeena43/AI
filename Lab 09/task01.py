import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_excel('houses.xlsx')

le = LabelEncoder()
df['neighborhood'] = le.fit_transform(df['neighborhood'])

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_house = pd.DataFrame([[2200, 3, 2, 8, le.transform(['A'])[0]]], columns=X.columns)
predicted_price = model.predict(new_house)
print(f"Predicted Price: {predicted_price[0]}")
