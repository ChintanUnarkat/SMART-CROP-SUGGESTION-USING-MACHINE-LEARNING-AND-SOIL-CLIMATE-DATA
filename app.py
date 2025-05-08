from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
rf_model = None
svm_model = None
nb_model = None
knn_model = None

# Load the trained models when the app starts
try:
    rf_model = joblib.load('crop_model.pkl')
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print("Error: crop_model.pkl not found!")

try:
    svm_model = joblib.load('svm_crop_model.pkl')
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: svm_crop_model.pkl not found!")

try:
    nb_model = joblib.load('naive_bayes_crop_model.pkl')
    print("Naive Bayes model loaded successfully.")
except FileNotFoundError:
    print("Error: naive_bayes_crop_model.pkl not found!")

try:
    knn_model = joblib.load('knn_crop_model.pkl')
    print("KNN model loaded successfully.")
except FileNotFoundError:
    print("Error: knn_crop_model.pkl not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['model']
    prediction = "Error: Model not selected or loaded."

    # Corrected line to handle the generator object
    input_features = [float(x) for x in list(request.form.values())[1:]]
    final_features = [np.array(input_features)]

    if selected_model == 'rf':
        if rf_model:
            try:
                prediction = rf_model.predict(final_features)[0]
            except Exception as e:
                prediction = f"Error during Random Forest prediction: {e}"
        else:
            prediction = "Error: Random Forest model not loaded."
    elif selected_model == 'svm':
        if svm_model:
            try:
                prediction = svm_model.predict(final_features)[0]
            except Exception as e:
                prediction = f"Error during SVM prediction: {e}"
        else:
            prediction = "Error: SVM model not loaded."
    elif selected_model == 'nb':
        if nb_model:
            try:
                prediction = nb_model.predict(final_features)[0]
            except Exception as e:
                prediction = f"Error during Naive Bayes prediction: {e}"
        else:
            prediction = "Error: Naive Bayes model not loaded."
    elif selected_model == 'knn':
        if knn_model:
            try:
                prediction = knn_model.predict(final_features)[0]
            except Exception as e:
                prediction = f"Error during KNN prediction: {e}"
        else:
            prediction = "Error: KNN model not loaded."

    return render_template('index.html', prediction=prediction, selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=False) # Keep debug=False for Windows