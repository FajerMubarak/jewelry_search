# 💎 Khazeena AI — Visual Search Engine

A lightweight, production-grade visual search engine for jewelry items. Powered by **MobileNetV2 (ONNX)** and **Cosine Similarity**, this application retrieves visually similar jewelry products from a catalog based on an uploaded image or real-time camera snapshot.

---

## 🔗 Quick Links

- **🚀 Live Demo App:** [Click here to view the app](YOUR_STREAMLIT_APP_URL_HERE)
- **📊 Dataset Source:** [Tanishq Jewellery Dataset on Kaggle](https://www.kaggle.com/datasets/sapnilpatel/tanishq-jewellery-dataset)

---

## 🚀 Features

- **Multi-input Options**: Upload an image (`.jpg`, `.png`) or capture a photo using your camera.
- **Fast Inference with ONNX**: Utilizes `onnxruntime` for fast, lightweight neural network feature extraction without heavy AI framework dependencies.
- **L2 Normalized Search**: Matches images using Cosine Similarity on feature embeddings with high accuracy (0% - 100%).
- **Clean Responsive UI**: Built with Streamlit, optimized for high-performance visual catalog browsing.
