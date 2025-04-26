import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

def read_and_prepare_data(path):
    data = pd.read_excel(path)
    data.fillna({
        'total_spent': data['total_spent'].median(),
        'num_of_visits': data['num_of_visits'].median(),
        'purchase_frequency': data['purchase_frequency'].median()
    }, inplace=True)

    for feature in ['total_spent', 'num_of_visits', 'purchase_frequency']:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        data = data[(data[feature] >= q1 - 1.5 * iqr) & (data[feature] <= q3 + 1.5 * iqr)]

    return data

def process_features(data):
    features = data[['total_spent', 'num_of_visits', 'purchase_frequency']]
    labels = data['value']

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return features_scaled, labels, scaler

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    classifier = SVC(kernel='linear', C=1.0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    features_2d = features[['total_spent', 'num_of_visits']]
    labels_2d = labels

    visual_classifier = SVC(kernel='linear', C=1.0)
    visual_classifier.fit(features_2d, labels_2d)

    DecisionBoundaryDisplay.from_estimator(
        visual_classifier, features_2d, response_method="predict",
        plot_method="pcolormesh", alpha=0.3
    )
    plt.scatter(features_2d.iloc[:, 0], features_2d.iloc[:, 1], c=labels_2d, edgecolors='k')
    plt.xlabel('Total Spent (scaled)')
    plt.ylabel('Number of Visits (scaled)')
    plt.title('Decision Boundary Visualization')
    plt.show()

    return classifier

def display_classification_rules(classifier, scaler):
    weights = classifier.coef_[0]
    bias = classifier.intercept_[0]

    print("\nDecision Function:")
    print(f"{weights[0]:.2f}*(total_spent) + {weights[1]:.2f}*(num_of_visits) + {weights[2]:.2f}*(purchase_frequency) + {bias:.2f} = 0")
    print("Classify as high-value if the result is greater than 0.")

def predict_customer(classifier, scaler, input_features):
    input_scaled = pd.DataFrame(scaler.transform(input_features), columns=input_features.columns)
    prediction = classifier.predict(input_scaled)
    result = 'HIGH VALUE' if prediction[0] == 1 else 'LOW VALUE'
    print(f"\nPrediction for new customer: {result}")

if __name__ == "__main__":
    dataset = read_and_prepare_data('customers.xlsx')
    features, labels, scaler = process_features(dataset)
    model = train_model(features, labels)
    display_classification_rules(model, scaler)

    sample_customer = pd.DataFrame({
        'total_spent': [500],
        'num_of_visits': [8],
        'purchase_frequency': [3]
    })
    predict_customer(model, scaler, sample_customer)
