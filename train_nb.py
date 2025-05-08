import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('Crop_recommendation.csv')

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']  # Assuming 'label' is the column with crop names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy on Test Set: {accuracy:.4f}")

joblib.dump(nb_model, 'naive_bayes_crop_model.pkl')
print("Naive Bayes model saved as naive_bayes_crop_model.pkl")