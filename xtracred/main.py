import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 
plt.rcParams['figure.figsize'] = 15, 10
  
# This code was a modified version of the tutorial from GeeksForGeeks
# Link: https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/

# Create custom transformation and normalize the image
transform = torchvision.transforms.Compose([ 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5), (0.5)) 
]) 
  
# Load the dataset from torchvision. Will automatically download the dataset if not existed
train_dataset = torchvision.datasets.MNIST( 
    root="./MNIST/train", train=True, 
    transform=torchvision.transforms.ToTensor(), 
    download=True) 
  
test_dataset = torchvision.datasets.MNIST( 
    root="./MNIST/test", train=False, 
    transform=torchvision.transforms.ToTensor(), 
    download=True) 
  
# Data transformation and create dataloader for easy use later
train_loader = torch.utils.data.DataLoader( 
    train_dataset, batch_size=256) 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, batch_size=256) 

# Model specification
model = utils.generate_model() 
criterion = nn.MSELoss() 
num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

# Train Loss over time
train_loss = [] 
  
# Output state per batch
outputs = {} 
  
batch_size = len(train_loader) 
  
# Training
for epoch in range(num_epochs): 
    running_loss = 0
    for batch in train_loader: 
        img, _ = batch   
        img = img.reshape(-1, 28*28) 
        out = model(img) 
        loss = criterion(out, img) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item() 
    running_loss /= batch_size 
    train_loss.append(running_loss) 
    outputs[epoch+1] = {'img': img, 'out': out} 
  
  
# Plotting the training loss 
plt.plot(range(1,num_epochs+1),train_loss) 
plt.xlabel("Number of epochs") 
plt.ylabel("Training Loss") 
plt.show()
  
outputs = {} 

# Experimental and Analyze
# Grab an image from the test_loader
# Get the noise extract using Gaussian blur
# Using solver function from utils to solve the optimization problem
# Plot both the original and pertubed images
originalImageRaw = list(test_loader)[-1][0][9]
originalImage = originalImageRaw.reshape(-1,28*28)
out1 = model(originalImage)
out1Img = out1.detach().numpy()
out1up = out1Img.reshape(28,28)
noise = cv2.GaussianBlur(np.squeeze(originalImageRaw.numpy()), (19,19), 5)
noiseExtract = noise - np.squeeze(originalImageRaw.numpy())
result = utils.solver(originalImageRaw, noiseExtract, model)
adv = np.squeeze(originalImageRaw.numpy()) + result.x*noiseExtract
noiseExp = np.expand_dims(adv, axis=0)
noiseTensor = torch.Tensor(noiseExp)
noiseImg = noiseTensor.reshape(-1,28*28)
output = model(noiseImg)
out2Img = output.detach().numpy()
outputToshow = out2Img.reshape(28,28)
inputToshow = originalImage.reshape(28,28)
advToshow = noiseImg.reshape(28,28)
print(result.x)
plt.imshow(inputToshow, cmap='gray')
plt.show()
plt.imshow(out1up, cmap='gray')
plt.show()
plt.imshow(advToshow,cmap='gray')
plt.show()
plt.imshow(outputToshow, cmap='gray') 
plt.show()