# 🛡️ Kernel SVM Dashboard — Social Network Ads

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)



[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?logo=rocket)](https://kernelsvmdashboard.streamlit.app/)

</p>

---

## 🚀 Overview

An interactive **Machine Learning Dashboard** that demonstrates how a **Kernel Support Vector Machine (SVM)** predicts customer purchasing behavior based on:

* 📅 Age
* 💰 Estimated Salary

The app visualizes **decision boundaries**, evaluates model performance, and allows real-time predictions.

---

## ✨ Features

* 🧠 Train an **RBF Kernel SVM model**
* 🎛️ Adjust hyperparameters (C, test size)
* 📊 View **accuracy & confusion matrix**
* 📈 Interactive **decision boundary visualization (Plotly)**
* 🔍 Real-time **user input prediction**
* ⚡ Fully interactive UI with Streamlit

---

## 🛠️ Tech Stack

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly

---

## 📂 Project Structure

```bash
.
├── app.py
├── Social_Network_Ads.csv
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/kernel-svm-dashboard.git
cd kernel-svm-dashboard
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 Click the badge above or visit:
https://your-streamlit-app-link.streamlit.app

---

## 📊 Model Details

* Algorithm: **Support Vector Machine (SVM)**
* Kernel: **Radial Basis Function (RBF)**
* Features:

  * Age
  * Estimated Salary
* Target:

  * Purchase Decision (0 / 1)

---

## 📈 Visualizations

* 🔵 Decision boundary (training & test sets)
* 🔴 Class separation regions
* 📊 Confusion matrix
* 📉 Model accuracy

---

## 🧠 How It Works

1. Data is loaded from `Social_Network_Ads.csv`
2. Split into training and test sets
3. Features are standardized
4. SVM model is trained using RBF kernel
5. Predictions are visualized using Plotly

---

## 🎯 Live Prediction

Use the sidebar to input:

* Age
* Salary

The model predicts whether the user is likely to **purchase or not**.

---

## 🚀 Deployment

Deploy easily using **Streamlit Cloud**:

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create new app
4. Upload repo & deploy

---

## 🔮 Future Improvements

* 📊 Add ROC Curve & AUC
* 📉 Compare with other models (Logistic Regression, KNN)
* 🧠 Hyperparameter tuning (GridSearchCV)
* 📁 Upload custom datasets
* 📊 Feature importance visualization

---

## 👨‍💻 Author

**Chinmay V Chatradamath.**

---

## 🤝 Contributing

Pull requests are welcome!

---

## 📄 License

MIT License

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
