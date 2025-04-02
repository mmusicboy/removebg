from download_model import download_model
download_model()

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Загрузка модели U^2-Net
from model.u2net import U2NET  # Импорт из модели

MODEL_PATH = 'saved_models/u2net/u2net.pth'

def load_model():
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    net.eval()
    return net

model = load_model()

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess(mask, original_size):
    mask = mask.squeeze().cpu().data.numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
    return mask

def remove_background(image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
    mask = postprocess(d1, image.size)
    empty = Image.new("RGBA", image.size, (0, 0, 0, 0))
    image = image.convert("RGBA")
    datas = image.getdata()
    masks = mask.getdata()

    new_data = []
    for item, m in zip(datas, masks):
        new_data.append((item[0], item[1], item[2], m))

    empty.putdata(new_data)
    return empty
