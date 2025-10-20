import numpy as np
import pandas as pd
from sklearn.datasets import load_iris   
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
data=pd.read_csv('iris.csv')
print("Iris Data loaded successfully.")
print(data.head())
# Separate features and target
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  
y = data['Species']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)    #predict test result
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)


