
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

diabetes_data = pd.read_csv('diabetes.csv')


print("Dataset Info:")
print(diabetes_data.info())

print("\nSummary Statistics:")
print(diabetes_data.describe())

plt.figure(figsize=(16, 12))
for i, column in enumerate(diabetes_data.columns[:-1], 1):  
    plt.subplot(3, 3, i)
    sns.histplot(diabetes_data[column], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
correlation_matrix = diabetes_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    diabetes_data[col] = diabetes_data[col].replace(0, np.nan)
    diabetes_data[col] = diabetes_data[col].fillna(diabetes_data[col].mean())


print("\nPreprocessed Data Preview:")
print(diabetes_data.head())


X = diabetes_data.drop(columns='Outcome')
y = diabetes_data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nScaled Features Preview:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"\nTraining Set Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Set Shape: {X_test.shape}, {y_test.shape}")


log_model = LogisticRegression()
log_model.fit(X_train, y_train)


y_pred_log = log_model.predict(X_test)


print("\n--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

conf_matrix_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest')
plt.show()

joblib.dump(rf_model, 'rf_model.pkl')


joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully!")
