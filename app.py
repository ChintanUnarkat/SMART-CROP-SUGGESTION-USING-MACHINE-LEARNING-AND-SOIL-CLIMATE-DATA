from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load all ML models
models = {
    "Random Forest": joblib.load("crop_model.pkl"),
    "SVM": joblib.load("svm_crop_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_crop_model.pkl"),
    "KNN": joblib.load("knn_crop_model.pkl")
}

# Descriptive information for each crop
crop_info = {
    "rice": "Rice is a staple crop that requires high humidity, warm temperatures, and flooded soil. It thrives in regions with heavy rainfall and is widely grown in Asia.",
    "maize": "Maize, or corn, grows well in moderate temperatures and sunlight. It’s used for food, fodder, and biofuel.",
    "cotton": "Cotton is a fiber crop that prefers warm climates, moderate rainfall, and loamy soils. It's used widely in the textile industry.",
    "wheat": "Wheat is a cereal crop that prefers cool climates and well-drained loamy soil. It is one of the most important food grains worldwide.",
    "sugarcane": "Sugarcane needs hot and humid conditions and is grown in regions with rich loamy soil and plenty of water.",
    "coffee": "Coffee is grown in cool-to-warm climates, usually in hilly regions. It prefers shade, good drainage, and high rainfall.",
    "banana": "Banana plants thrive in tropical regions with warm temperatures and rich, well-irrigated soil. They are sensitive to wind and drought.",
    "mango": "Mango trees require warm tropical weather and well-drained soil. They are sensitive to frost and require a dry period before flowering.",
    "grapes": "Grapes need a warm climate with low humidity and well-drained soil. They are often grown in vineyards with support systems.",
    "apple": "Apples require a temperate climate with cold winters and moderate summer temperatures. Grown best in hilly regions with rich soil.",
    "pigeonpeas": "Pigeonpeas are drought-resistant legumes that thrive in warm climates. They are often grown in poor soils and are used for food and fodder.",
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model_name = request.form.get('model')
        input_values = [float(x) for x in list(request.form.values())[1:]]
        input_array = np.array([input_values])

        if selected_model_name in models:
            model = models[selected_model_name]
            prediction = model.predict(input_array)[0]
            crop = prediction.lower()
            description = crop_info.get(crop, "No additional information available.")
            return render_template('index.html', prediction=prediction, description=description)
        else:
            return render_template('index.html', prediction="⚠️ Model not found.", description="")
    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {str(e)}", description="")

if __name__ == '__main__':
    app.run(debug=True)
