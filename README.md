# Remove Background API with U²-Net

This is a Flask-based API that removes the background from uploaded images using U²-Net, deployed on Render.com.

## 🚀 How to use

- POST an image to `/remove-bg`
- Returns a PNG with a transparent background

## 🧠 Model

Uses the U²-Net model for human segmentation.

## Setup

```
pip install -r requirements.txt
python main.py
```
