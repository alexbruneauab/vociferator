import torch
from torch import nn

from model_neural_network import NeuralNetwork
from model_data import test_data

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# #Creating a new model based on the existing nn and load the model's dictionnary
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("tutorial_pytorch_qs/model_dict.pth"))

#Load the entire model
model = torch.load('tutorial_pytorch_qs/model.pth')

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')