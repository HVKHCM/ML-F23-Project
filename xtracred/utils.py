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


def frobenius_norm(originalImageRaw, noise, model, x):
    originalImage = originalImageRaw.reshape(-1,28*28)
    map1 = model(originalImage)
    adv = np.squeeze(originalImageRaw.numpy()) + x*noise
    noiseExp = np.expand_dims(adv, axis=0)
    noiseTensor = torch.Tensor(noiseExp)
    noiseImg = noiseTensor.reshape(-1,28*28)
    map2 = model(noiseImg)
    difference = map2 - map1
    difference = difference.reshape(-1,28*28)
    difference = difference.detach().numpy()
    norm = 0
    print(difference)
    for i in range(784):
            norm += difference[0][i]**2
    return np.sqrt(norm)

def objective(x,originalImageRaw, noise, model):
    return 1/frobenius_norm(originalImageRaw, noise, model, x)

def solver(originalImageRaw, noise, model):
    result = opt.minimize(objective,0.5,args=(originalImageRaw,noise,model,), bounds=[(0.01,1.0)])
    return result


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