<div align="center">

# 🌿 AgriScan

### AI-Powered Crop & Weed Classification System using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=flat-square)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-success?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

### 🌱 Detect whether a plant is a Crop or a Weed using Deep Learning

Upload a plant image and instantly receive an AI-powered prediction with confidence score and visual detection overlay.

</div>

---

# 📌 Overview

AgriScan is a deep-learning based agricultural image classification system designed to identify whether a plant belongs to the **Crop** or **Weed** category.

The system leverages a fine-tuned **YOLOv8** model trained on agricultural plant images and provides predictions through a simple and interactive **Streamlit web application**.

This project demonstrates the practical application of Computer Vision and Deep Learning in Smart Agriculture, helping automate crop monitoring and weed identification.

---

# 🚀 Features

✅ Crop Detection

✅ Weed Detection

✅ Confidence Score Visualization

✅ Bounding Box Detection Overlay

✅ Real-Time Image Prediction

✅ User-Friendly Streamlit Interface

✅ YOLOv8 Deep Learning Model

✅ Ready for Future Mobile Deployment

---

# 🖼️ Demo

### Home Screen

Add screenshot here:

```text
assets/homepage.png
```

### Crop Prediction

Add screenshot here:

```text
assets/crop_prediction.png
```

### Weed Prediction

Add screenshot here:

```text
assets/weed_prediction.png
```

---

# 📊 Model Performance

The YOLOv8 model was trained on a custom agricultural dataset containing crop and weed images.

| Metric    | Score |
| --------- | ----- |
| Precision | 80.6% |
| Recall    | 77.8% |
| mAP@50    | 85.8% |
| mAP@50-95 | 56.4% |

These results demonstrate strong classification capability while maintaining lightweight inference performance.

---

# 🏗️ Project Architecture

```text
                User Uploads Image
                         │
                         ▼
                Streamlit Web App
                         │
                         ▼
                 Image Preprocessing
                         │
                         ▼
                YOLOv8 Inference Model
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
       Crop                          Weed
          │                             │
          └──────────────┬──────────────┘
                         ▼
              Confidence Score
                         ▼
             Detection Visualization
```

---

# 📂 Project Structure

```text
AgriScan/
│
├── app.py
│
├── model/
│   └── agriscan_yolov8.pt
│
├── assets/
│   ├── homepage.png
│   ├── crop_prediction.png
│   └── weed_prediction.png
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── dataset/
    ├── train/
    ├── valid/
    └── test/
```

---

# ⚙️ Technologies Used

### Programming Language

* Python

### Deep Learning

* YOLOv8
* PyTorch

### Computer Vision

* OpenCV
* Pillow

### Web Framework

* Streamlit

### Model Training Platform

* Google Colab (Tesla T4 GPU)

---

# 🔥 Installation Guide

## Clone Repository

```bash
git clone https://github.com/your-username/agriscan.git

cd agriscan
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Add Trained Model

Place the trained YOLOv8 weights inside:

```text
model/agriscan_yolov8.pt
```

---

## Run Application

```bash
streamlit run app.py
```

The application will launch automatically in your browser:

```text
http://localhost:8501
```

---

# 🧠 Model Training

The model was trained using the Ultralytics YOLOv8 framework.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

---

# 📁 Dataset Format

```text
dataset/

├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

Classes:

```yaml
0: crop
1: weed
```

---

# 🔍 How Prediction Works

```text
Image Upload
      │
      ▼
Temporary Image Storage
      │
      ▼
YOLOv8 Model Prediction
      │
      ▼
Class Identification
      │
      ▼
Confidence Calculation
      │
      ▼
Bounding Box Rendering
      │
      ▼
Result Display
```

---

# 🌾 Agricultural Applications

AgriScan can be extended for:

* Precision Agriculture
* Weed Monitoring
* Crop Health Assessment
* Smart Farming Systems
* Automated Farm Surveillance
* Agricultural Robotics

---

# 🛣️ Future Enhancements

* [ ] Multi-Plant Detection
* [ ] Weed Density Estimation
* [ ] Mobile Application
* [ ] Real-Time Camera Detection
* [ ] Farm Drone Integration
* [ ] Disease Detection Module
* [ ] Crop Recommendation System
* [ ] Cloud Deployment

---

# 👨‍💻 Author

### Shadrack A

B.Tech Information Technology

St. Joseph's Institute of Technology, Chennai

Interests:

* Artificial Intelligence
* Machine Learning
* Data Science
* Computer Vision
* Smart Agriculture

GitHub:
https://github.com/Shad18

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify and distribute this project for educational and research purposes.

---

<div align="center">

### 🌱 AgriScan — Smart Agriculture Powered by Artificial Intelligence

Built with ❤️ using YOLOv8, PyTorch and Streamlit

</div>
