from model import ResNet
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose, Resize
import numpy as np
import matplotlib.pyplot as plt


model = ResNet()
size=64
five_sisters_name = ["itika", "nino", "miku", "yotuba", "ituki"]
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

model_path = f'F:/projects/five_sisters/weight/best_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

image_path = f'F:/projects/five_sisters/data_set/ituki/21.png'
image = read_image(image_path, ImageReadMode.RGB)
transform = Compose([Resize((size, size))])
image = transform(image)
image = image.reshape((1, 3, size, size))
image = image/255.0

with torch.no_grad():
    y = model(image).numpy()
y = softmax(y[0])

for i in range(5):
    five_sisters = five_sisters_name[i], y[i]

print(five_sisters_name[y.argmax()], y.max())
image = np.array(image)
image = image.reshape((3, size, size))
image = image.transpose((1, 2, 0))
plt.imshow(image)
plt.show()