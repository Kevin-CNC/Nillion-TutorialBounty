
# Main python file by Kevin Caplescu
# Following tutorial provided by Nillion to create & fine-tune decentralised AI models BerTiny & LeNet5

# Load the libraries
import aivm_client as aic
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# MNIST Dataset is loaded through function 'load_mnist'
def load_mnist():
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5,), (1.0,)),
        ]
    )
    train_set = dset.MNIST(
        root="/tmp/mnist", train=True, transform=trans, download=True
    )
    return train_set

dataset = load_mnist()

inputs, labels =  dataset[20]
inputs = inputs.reshape(1, 1, 28, 28)

# After loading dataset, we visualise the selected digits, verifying the data input
plt.imshow(inputs.detach().numpy().squeeze(), cmap='gray')
plt.title(f'Grayscale Image of a {labels}')
plt.show()

# We then make an actual inference to the LeNet5 AI Model.
encrypted_inputs = aic.LeNet5Cryptensor(inputs)
result = aic.get_prediction(encrypted_inputs, "LeNet5MNIST") # Predicted results here
results = torch.argmax(result, dim=1)
print("Predicted Label:", results.item()) 

# After getting the predicted results, we will process the actual images and compare to the predictions made, in order to assess accuracy
for i in range(100):
    inputs, labels =  dataset[i]
    inputs = inputs.reshape(1, 1, 28, 28)
    encrypted_inputs = aic.LeNet5Cryptensor(inputs)
    result = aic.get_prediction(encrypted_inputs, "LeNet5MNIST")
    results = torch.argmax(result, dim=1)
    print("Predicted Label:", results.item(), "True Label:", labels, "Correct:", results.item() == labels)
    

# BertTiny inference & input tokenization for SMS Spam detection
    
# Input is tokenized and encrypted
tokenized_inputs = aic.tokenize("Your free ringtone is waiting to be collected. Simply text the password 'MIX' to 85069 to verify. Get Usher and Britney. FML, PO Box 5249, MK17 92H. 450Ppw 16")
encrypted_inputs = aic.BertTinyCryptensor(*tokenized_inputs)

# A result is then predicted by the BertTiny model that has been pre-trained to SMS 
result = aic.get_prediction(encrypted_inputs, "BertTinySMS")
"SPAM" if torch.argmax(result) else "HAM"
