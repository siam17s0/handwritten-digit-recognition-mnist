import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Load trained model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# UI
st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    img = np.array(image)

    # threshold
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    # crop digit
    coords = cv2.findNonZero(img)
    x,y,w,h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # resize digit
    img = cv2.resize(img, (20,20))

    # create canvas
    canvas = np.zeros((28,28))
    canvas[4:24,4:24] = img

    img = canvas

    # normalize
    img = img / 255.0
    img = (img - 0.5) / 0.5

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    # prediction
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    st.success(f"Predicted Digit: {pred.item()}")
