import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Imag
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import aivm_client as aic # Import the Nillion-AIVM client


# Checking Hardware availability for training
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
    
# Load official microsoft dataset for cats and dogs comparison
ds = load_dataset("microsoft/cats_vs_dogs")
def plot_images(dataset, num_images=4):
  
  # Plots a specified number of images from the dataset.
  fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
  for i in range(num_images):
    image = dataset['train'][i]['image']
    image = np.array(image)
    axs[i].imshow(image)
    axs[i].axis('off')
    axs[i].set_title(f"Label: {dataset['train'][i]['labels']}")
  plt.show()

plot_images(ds)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.5,), (1.0,))
])

# Apply transformation to the dataset
def transform_dataset(example):
    # Apply the transform on the 'image' field
    image = example['image']
    image = transform(image)
    example['image'] = image
    return example

# Apply transform to the dataset
ds = ds.with_format("torch")
ds = ds.map(transform_dataset)

ds['train'] = ds['train'].shuffle(seed=42)
ds = ds['train'].train_test_split(test_size=0.1)

train_loader = torch.utils.data.DataLoader([(x, y) for x,y in zip(ds['train']['image'], ds['train']['labels'])], batch_size=32)
test_loader = torch.utils.data.DataLoader([(x, y) for x,y in zip(ds['test']['image'], ds['test']['labels'])], batch_size=32)


def plot_transformed_images(dataset, num_images=4):
  """Plots a specified number of images from the transformed dataset."""
  fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
  for i in range(num_images):
    image = dataset['train'][i]['image'].permute(1, 2, 0).numpy()  # Move channel dimension to the end
    axs[i].imshow(image, cmap='gray')  # Display grayscale image
    axs[i].axis('off')
    axs[i].set_title(f"Label: {dataset['train'][i]['labels']}")
  plt.show()


plot_transformed_images(ds)

# Definining the LeNet5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Class LeNet5 constructed in order to define architecture.
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14

            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5

        )
        self.flattener = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


network = LeNet5()
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=1e-3)

# Instantiate a torch loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop function for the LeNet5 Model through N number of epochs.
def Training_Function(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = network(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    network.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            #print(data.shape)
            logits = network(data)
            train_loss += loss_fn(logits, target).item()
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    train_loss /= len(train_loader.dataset)
    print('\nTraining set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
      #torch.save(network.state_dict(), '/results/model.pth')
      #torch.save(optimizer.state_dict(), '/results/optimizer.pth')
      
# Model is switched into 'evaluation mode', where the data is now passed through the model and evaluated for potential predictions.
def Assessment_Mode():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  

N_EPOCHS = 5
for epoch in range(1, N_EPOCHS + 1):
  Training_Function(epoch)
  Assessment_Mode()
  
# Trained model is then saved as a pth file.
torch.save(network.to('cpu').state_dict(), "./cats_vs_dogs_lenet5.pth")

# After training, assessing and saving the model, we will then upload the LeNet5 model.
MODEL_NAME = "My-LeNet5Cats&Doggos-Model" # Name of the model to be used
aic.upload_lenet5_model("./cats_dogs_lenet5.pth", MODEL_NAME) # Upload the model to the server

# Define transformation: Resize to 28x28 and convert to grayscale
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.5,), (1.0,))
])

# We now proceed to make an encrypted inference with the current trained LeNet5 AI.
img_tensor = transform(ds["train"][1]["image"])
encrypted_input = aic.LeNet5Cryptensor(img_tensor.reshape(1, 1, 28, 28))

prediction = aic.get_prediction(encrypted_input, MODEL_NAME)
print("CAT" if torch.argmax(prediction).item() == 0 else "DOG")