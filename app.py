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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()


# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0-9)")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # Convert to numpy
    img = np.array(image)

    # Resize to MNIST size
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    # Invert if background is white
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize like training
    img = img / 255.0
    img = (img - 0.5) / 0.5

    # Show processed image
    st.image(img, caption="Processed Image (28x28)", width=150)

    # Convert to tensor
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs,1)

    # Show result
    st.success(f"Predicted Digit: {predicted.item()}")

    st.subheader("Prediction Probabilities")
    probs = probabilities.numpy()[0]

    for i,p in enumerate(probs):
        st.write(f"{i} : {p:.4f}")

    st.bar_chart(probs)
