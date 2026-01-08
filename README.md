# 🂡 Playing Card Image Classifier (PyTorch + FastAPI)

A deep learning–based image classification project that identifies playing cards from images using **PyTorch** and **EfficientNet-B0**, served through a **FastAPI** backend. The model predicts the card category along with a confidence score and achieves **~95% accuracy**.

---

## 🚀 Live Demo (API)
The API is live and accessible here:

🔗 https://pytorch-card-image-classifier.onrender.com

Upload a card image via the web interface to get instant predictions.

---

## 📌 Project Highlights
- Deep learning model built with **PyTorch**
- Transfer learning using **EfficientNet-B0**
- Fast and lightweight **FastAPI** backend
- Dockerized for easy deployment
- Predicts card category + confidence score
- Achieves approximately **95% accuracy**

---

## 🧠 Model & Dataset
- Dataset: Playing card images (53 classes including Joker)
- Training approach: Transfer learning
- Output: Card name (e.g., *ACE OF SPADES*) with confidence %

⚠️ **Important Note**  
The dataset is available only on **Kaggle**, so training should be run **only on Kaggle notebooks**.

📓 Kaggle Training Notebook:  
https://www.kaggle.com/code/almusheer/pytorch-card-classification

---

## 🛠️ How to Use (API)

### Run using Docker
```bash
docker run -p 8000:8000 musheer/playing-card-classifier-api


Open in browser
http://localhost:8000

API Endpoint
POST /predict


Upload an image file of a playing card

Receive:

Predicted card category

Confidence score

📦 Tech Stack

Python

PyTorch

Torchvision

EfficientNet (timm)

FastAPI

Docker

🎯 Use Cases

Image classification demos

Learning PyTorch + FastAPI integration

College / academic projects

ML model deployment practice

Portfolio project

👨‍💻 Author

Mohd Musheer

If you find this project useful, feel free to ⭐ the repository.