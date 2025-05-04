import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and clean the dataset
df = pd.read_excel('customers.xlsx')

# Fill missing values with median
df.fillna({
    'total_spent': df['total_spent'].median(),
    'num_of_visits': df['num_of_visits'].median(),
    'purchase_frequency': df['purchase_frequency'].median()
}, inplace=True)

# Remove outliers using IQR
for col in ['total_spent', 'num_of_visits', 'purchase_frequency']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# 2. Prepare features and labels
X = df[['total_spent', 'num_of_visits', 'purchase_frequency']]
y = df['value']

# 3. Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train SVC model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Display hyperplane
weights = model.coef_[0]
bias = model.intercept_[0]
print("\nHyperplane Equation:")
print(f"{weights[0]:.2f}*(total_spent) + {weights[1]:.2f}*(num_of_visits) + {weights[2]:.2f}*(purchase_frequency) + {bias:.2f} = 0")

# 8. Plot decision boundary (2D using total_spent and num_of_visits)
X_2d = X_scaled[['total_spent', 'num_of_visits']]
model_2d = SVC(kernel='linear', C=1.0)
model_2d.fit(X_2d, y)

# Create mesh grid for plotting
x_min, x_max = X_2d['total_spent'].min() - 1, X_2d['total_spent'].max() + 1
y_min, y_max = X_2d['num_of_visits'].min() - 1, X_2d['num_of_visits'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_2d.predict(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d['total_spent'], X_2d['num_of_visits'], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('Total Spent (scaled)')
plt.ylabel('Number of Visits (scaled)')
plt.title('Decision Boundary (SVC - 2D View)')
plt.grid(True)
plt.show()

# 9. Predict new customer
new_customer = pd.DataFrame([[500, 8, 3]], columns=X.columns)
new_customer_scaled = pd.DataFrame(scaler.transform(new_customer), columns=X.columns)
prediction = model.predict(new_customer_scaled)
result = 'HIGH VALUE' if prediction[0] == 1 else 'LOW VALUE'
print(f"\nPrediction for new customer: {result}")
