import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import utils
from mpl_toolkits import mplot3d



eigenvectors, eigenvalues, plot = utils.pca3d("../data1.csv")

plt.show()

step = utils.ratioProgress("../data1.csv")
step.show()

tsne = utils.tsneAnalysis("../data1.csv")
tsne.show()