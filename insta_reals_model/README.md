# 📈 Instagram Video Virality Prediction (ML Project)

This project explores how **Machine Learning (ML)** can be used to **analyze and predict the potential virality of Instagram videos** before posting.  
It’s inspired by how **Instagram’s own recommendation algorithms** work — using a combination of **computer vision, NLP, and engagement prediction models**.

---

## 🚀 Objective
To estimate a *"Virality Score"* for short-form video content (like Reels), based on:
- Visual quality (composition, motion, brightness)
- Caption sentiment and keyword strength
- Predicted engagement metrics (likes, comments, saves)
- Audio or hashtag trends

The goal: **help creators test and optimize videos before uploading**.

---

## 🧠 Model Overview
This project combines multiple open-source machine learning models:

| Component | Model | Description |
|------------|--------|-------------|
| 🖼️ Visual Quality | [NIMA (Neural Image Assessment)](https://github.com/tensorflow/models/tree/master/research/nima) | Rates the aesthetic appeal of images and video frames |
| 🎥 Video Quality | [VSFA (Video Quality Assessment)](https://github.com/lidq92/VSFA) | Evaluates motion smoothness and overall video clarity |
| 🧾 Text Sentiment | [BERT / RoBERTa](https://github.com/huggingface/transformers) | Analyzes caption tone and engagement-related keywords |
| ❤️ Virality Prediction | [TikTok Virality Model](https://github.com/harbarex/tiktok-virality-prediction) | Predicts short-form video popularity using deep learning |
| 🧮 Aggregator | Custom Scoring Model | Combines multiple signals into one “Virality Score” (0–100) |

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **PyTorch / TensorFlow**
- **OpenCV** (for frame extraction)
- **Transformers (Hugging Face)**
- **Pandas / NumPy / Scikit-Learn**
- **Matplotlib / Seaborn (for visualization)**

---

## 📊 Workflow

```mermaid
flowchart TD
A[Upload Video + Caption] --> B[Extract Frames]
B --> C[Visual Quality Model (NIMA)]
B --> D[Video Quality Model (VSFA)]
A --> E[Caption Sentiment Model (BERT)]
C & D & E --> F[Feature Aggregation + Virality Scoring]
F --> G[Output: Predicted Engagement / Virality Score]
