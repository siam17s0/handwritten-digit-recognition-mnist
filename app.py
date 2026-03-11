import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# CNN model architecture (same as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3*3*128 , 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
        

# load model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Handwritten Digit Recognition")

st.write("Upload a digit image (0-9)")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    img = np.array(image)

    img = cv2.resize(img,(28,28))

    img = img/255.0

    img = img.reshape(1,1,28,28)

    img = torch.tensor(img).float()

    with torch.no_grad():
        outputs = model(img)

        _, predicted = torch.max(outputs,1)

    st.success(f"Predicted Digit: {predicted.item()}")