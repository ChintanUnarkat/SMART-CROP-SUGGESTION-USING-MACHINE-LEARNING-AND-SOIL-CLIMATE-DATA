# SMART-CROP-SUGGESTION-USING-MACHINE-LEARNING-AND-SOIL-CLIMATE-DATA
This project builds a smart crop suggestion system using ML models like RF, KNN, SVM &amp; Naïve Bayes to predict suitable crops based on soil nutrients (N, P, K), pH, temperature, humidity, and rainfall, aiding farmers in making data-driven decisions.
## 📌 Abstract

This project aims to develop a smart crop suggestion system using machine learning techniques based on soil and climatic conditions. The goal is to assist farmers and agriculture stakeholders in selecting the most suitable crop by analyzing environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), pH, temperature, humidity, and rainfall. The proposed system leverages machine learning models like Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Naïve Bayes for accurate prediction.

A dataset containing labeled crop data was used to train and test the model. The system was then integrated with a user-friendly web interface developed using Flask, allowing users to input real-time environmental conditions and get crop recommendations. The proposed model achieved high accuracy and serves as a practical tool to support decision-making in precision agriculture.

---

## 📘 Introduction

In agriculture, choosing the right crop based on soil and environmental factors plays a critical role in increasing productivity and profitability. Traditional farming decisions are often based on experience, which may not always result in optimal outcomes. This project introduces a machine learning-based system that recommends suitable crops by analyzing environmental data.

The system uses historical agricultural data and machine learning algorithms to make predictions. Farmers can use this tool to make data-driven decisions that lead to improved yields and sustainable farming practices.

---

## 📚 Literature Survey (with Research Papers)

### 🔹 Paper 1: *Crop Recommendation System*  
- **Author:** Thewahettige Harinditha Ruchirawya  
- **Institute:** Sri Lanka Institute of Information Technology  
- **Published:** October 2020, International Journal of Computer Applications  
- **Link:** [Paper 1 PDF](https://drive.google.com/file/d/1X_FQcrkhJ7xONUC-Ce_oef4vo_wXLFlJ/view?usp=drivesdk)

**Summary:**  
This paper presents a Crop Recommendation System that uses real-time environmental data collected through Arduino-based sensors. Models like Naïve Bayes and SVM were used, along with NLP for farmer feedback. The system achieved ~92% accuracy and included a mobile app deployment.

---

### 🔹 Paper 2: *A Machine Learning Approach to Crop Recommendations*  
- **Authors:** Farida Siddiqi Prity, Md. Mehedi Hasan, et al.  
- **Published:** September 2024, Springer  
- **Link:** [Paper 2 PDF](https://drive.google.com/file/d/1Xb23BCmbi2w1SuKyxONoNNjM1FN9zmm7/view?usp=drivesdk)

**Summary:**  
This study tested nine ML algorithms (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, Extra Trees) for crop prediction. Features included N, P, K, pH, humidity, etc. Evaluation used precision, recall, F1-score, and ROC-AUC. Random Forest and Extra Trees performed best.

---

## 🧱 System Architecture

The system is divided into three main components:

- **Frontend**: A web interface where users enter environmental parameters  
- **Backend**: A Flask application that handles requests and interacts with the ML model  
- **Model**: A trained ML model that predicts the suitable crop based on inputs  

---

## 🧪 Methodology

- **Dataset**: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/karthikeyaagr/crop-recommendation-dataset)  
- **Features**: Nitrogen, Phosphorus, Potassium, pH, Temperature, Humidity, Rainfall  
- **Target**: Crop Label  

### ✅ Algorithms Used
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naïve Bayes  

### 🔧 Tools & Tech Stack
- Python, Pandas, Scikit-learn  
- Flask for backend  
- HTML/CSS for frontend  

The dataset was cleaned, normalized, and split into training/testing sets. The Random Forest model was selected for deployment due to its superior accuracy and speed.

---

## 💻 Implementation

The model was trained using Scikit-learn in Python. After evaluating multiple models, the Random Forest Classifier was selected. The trained model was saved using `joblib` and integrated into a Flask backend. The frontend accepts user input and displays the predicted crop.

---

## 📈 Results and Evaluation

| Model           | Accuracy | Log Loss   |
|----------------|----------|------------|
| Random Forest  | 0.9932   | 0.0628     |
| Naive Bayes    | 0.9955   | 0.0165     |
| SVM            | 0.9795   | –          |
| KNN            | 0.9705   | –          |

**Evaluation Metrics Used:**
- Confusion Matrix  
- Precision  
- Recall  
- F1-Score  
- Log Loss

The system accurately predicted crops across diverse environmental inputs.

---

## ✅ Conclusion and Future Scope

This project demonstrates the application of machine learning in precision agriculture. By analyzing soil and climate data, the system helps farmers make informed decisions and improve yields.

### 🔮 Future Enhancements:
- Integrate real-time weather APIs  
- Add soil type and market price analysis  
- Use sensors and microcontrollers for live data  
- Build a mobile app version  
- Add voice/chatbot assistant

---

## 🔗 References

1. [Crop Recommendation System – IJCA, 2020](https://drive.google.com/file/d/1X_FQcrkhJ7xONUC-Ce_oef4vo_wXLFlJ/view?usp=drivesdk)  
2. [ML Approach to Crop Recommendations – Springer, 2024](https://drive.google.com/file/d/1Xb23BCmbi2w1SuKyxONoNNjM1FN9zmm7/view?usp=drivesdk)  
3. [Kaggle Dataset – Crop Recommendation](https://www.kaggle.com/datasets/karthikeyaagr/crop-recommendation-dataset)  
4. [Scikit-learn Documentation](https://scikit-learn.org)

---

## 📂 Project Status

✅ Model Trained  
✅ Website Functional  
🛠️ Deployment In Progress  
