import torch 
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def simp_norm(map1, map2):
    difference = map2 - map1
    norm = 0
    for i in range(28):
        for j in range(28):
            norm += difference[i][j]**2
    return np.sqrt(norm)


def frobenius_norm(originalImage, noise, model, x):
    map1 = model(originalImage)
    map2 = model(originalImage + x*noise)
    difference = map2 - map1
    norm = 0
    for i in range(28):
        for j in range(28):
            norm += difference[i][j]**2
    return np.sqrt(norm)

def objective(originalImage, noise, model, x):
    return 1/frobenius_norm(originalImage, noise, model, x)

def adversarial(originalImage, noise, model):
    result = opt.minimize(objective, bounds=[(0,1)], args=(originalImage, noise, model, ))
    out = model(originalImage + result.x*noise)
    p = plt.imshow(out.reshape(28.28), cmap='gray')
    return result.x, p

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

#def creating_noise(originalImage):
#    transform = tv.transforms.Compose([transforms.ToTensor()])

class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(28 * 28, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 10) 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(10, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 28 * 28), 
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded 
  
# Instantiating the model and hyperparameters 
def generate_model():
    model = DeepAutoencoder()
    return model