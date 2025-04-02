import os
import requests

MODEL_DIR = "saved_models/u2net"
MODEL_PATH = os.path.join(MODEL_DIR, "u2net.pth")
MODEL_URL = "https://huggingface.co/flashingtt/U-2-Net/resolve/main/u2net.pth"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already downloaded.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")
