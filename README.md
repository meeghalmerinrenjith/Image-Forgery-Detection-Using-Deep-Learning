# 🛡️ Image Forgery Detection Using Deep Learning

A web-based deep learning application that detects whether a digital image is **Authentic** or **Forged** using a **Dual-Stream MobileNetV2** architecture and **Error Level Analysis (ELA)**. The project combines image preprocessing, transfer learning, and a Flask-based web interface to provide accurate image forgery detection.

---

## 📌 Project Overview

Digital image manipulation has become increasingly common with advanced editing tools, making it difficult to distinguish authentic images from forged ones. This project addresses this challenge by combining the original image with its Error Level Analysis (ELA) representation to improve forgery detection.

The model is trained on the **CASIA v1** and **CASIA v2** datasets using Transfer Learning with MobileNetV2 and deployed through a Flask web application.

---

## ✨ Key Features

- 🔍 Detects Authentic and Forged images
- 🖼️ Error Level Analysis (ELA) preprocessing
- 🤖 Dual-Stream MobileNetV2 Deep Learning model
- 📤 Single and Batch Image Upload
- 📊 Prediction with Confidence Score
- 🌐 Interactive Flask Web Application
- 📈 Model Evaluation using Confusion Matrix and Classification Report

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn
- HTML
- CSS
- Bootstrap

---

## 📂 Dataset

The model was trained using:

- CASIA v1 Image Tampering Detection Dataset
- CASIA v2 Image Tampering Detection Dataset

> **Note:** The datasets are not included in this repository due to their licensing terms.

---

## ⚙️ Project Workflow

```
CASIA Dataset
      │
      ▼
Dataset Preparation
      │
      ▼
Error Level Analysis (ELA)
      │
      ▼
Dual-Stream MobileNetV2 Training
      │
      ▼
Model Evaluation
      │
      ▼
Flask Web Application
```

---

## 📁 Project Structure

```
├── app.py
├── train_model.py
├── prepare_dataset.py
├── evaluate.py
├── requirements.txt
├── templates/
├── static/
├── README.md
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/your-username/Image-Forgery-Detection-Using-Deep-Learning.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000/
```

---

## 📊 Results

The model predicts whether an uploaded image is **Authentic** or **Forged** and provides a confidence score.

Performance evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📷 Project Images

This repository includes:

- Home Page
- Prediction Result
- Data Flow Diagram (DFD)
- UML Diagram
- Confusion Matrix

---

## 🔮 Future Scope

- Support additional forgery techniques
- Explainable AI (Grad-CAM)
- Video forgery detection
- Cloud deployment
- Vision Transformer (ViT)-based models

---

## 💼 Internship

This project was developed as part of my internship at **Akira Software Solutions Pvt. Ltd.**, in association with **Akira Knowledge Hub Pvt. Ltd.**, under the guidance of **Ms. Haritha H Pillai**.

---

## 👨‍💻 Author

**Meeghal Merin**

B.Tech Computer Science and Engineering (Data Science)

---

⭐ If you found this project interesting, consider giving it a star!
