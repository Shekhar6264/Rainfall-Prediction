# 🌧️ Rainfall Prediction Project

## 📌 Project Overview

This project is a **Rainfall Prediction System** built using **Machine Learning**. The dataset is trained with features like temperature, humidity, wind speed, and other weather-related parameters to predict rainfall. The trained model is saved as a `.pkl` file and can be used for predictions.

## 📂 Project Files

* `Rainfall.csv` → Dataset file used for training/testing.
* `Rainfall_Prediction.ipynb` → Jupyter Notebook containing data preprocessing, visualization, model training & evaluation.
* `rainfall_prediction_model.pkl` → Trained Machine Learning model saved for future predictions.

## ⚙️ Installation & Requirements

To run this project locally, follow these steps:

```bash
# 1️⃣ Clone the Repository
git clone https://github.com/Shekhar6264/Rainfall-Prediction.git
cd Rainfall-Prediction

# 2️⃣ Create Virtual Environment (optional but recommended)
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (Linux/Mac)
source venv/bin/activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# If requirements.txt is not available, install manually:
pip install numpy pandas matplotlib seaborn scikit-learn
```

## ▶️ Usage

Open the Jupyter Notebook to run and test the model:

```bash
jupyter notebook Rainfall_Prediction.ipynb
```

Or use the trained model (`rainfall_prediction_model.pkl`) in your Python script:

```python
import pickle
import numpy as np

# Load the trained model
with open('rainfall_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input (replace with actual features)
X_new = np.array([[25.3, 80, 12]])  # [Temperature, Humidity, WindSpeed]
prediction = model.predict(X_new)

print("🌧️ Rainfall Prediction:", prediction)
```

## 📊 Features

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning Model Training
* Model Evaluation & Accuracy
* Saved Model for Future Predictions

## 🚀 Future Improvements

* Add more features for better accuracy
* Deploy as a Web App (Flask/Django + React)
* Integrate with real-time weather APIs

## 👨‍💻 Author

**Shekhar Jamalpuri**
📧 Email: [shekharjamalpuri18@gmail.com](mailto:shekharjamalpuri18@gmail.com)
🔗 GitHub: [Shekhar6264](https://github.com/Shekhar6264)
