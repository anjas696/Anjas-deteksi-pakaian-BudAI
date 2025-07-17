import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn


# --- Model Dummy (Harus Anda ganti dengan model SSD Anda) ---
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- Load Model ---
model = DummyModel()
model.load_state_dict(torch.load('model/model_budAI100 (1).pth', map_location=torch.device('cpu')))
model.eval()


# --- Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --- Label Kelas ---
label_map = {
    0: "Wanita Tidak Berbudaya Islam",
    1: "Wanita Berpakaian Sesuai Budaya Islam"
}


# --- Tampilan Streamlit ---
st.set_page_config(page_title="Deteksi Pakaian Berbudaya Islam", layout="wide")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Deteksi Pakaian Berbudaya Islam (Real-Time)</h1>", unsafe_allow_html=True)

st.markdown(
"""
<p style="text-align: center;">Aplikasi ini digunakan untuk mendeteksi apakah seseorang berpakaian sesuai budaya Islam atau tidak secara real-time melalui webcam.</p>
""",
unsafe_allow_html=True
)


# --- Tombol Start Webcam ---
start_button = st.button('Mulai Deteksi Kamera')

FRAME_WINDOW = st.image([])

if start_button:
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            st.write("Kamera tidak dapat diakses.")
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = transform(pil_img).unsqueeze(0)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = label_map[predicted.item()]

        cv2.putText(img_rgb, pred_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        FRAME_WINDOW.image(img_rgb)

    camera.release()
