import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import utils
from mpl_toolkits import mplot3d



eigenvectors, eigenvalues, plot = utils.pca3d("../data1.csv")
plt.figure()
plt.savefig("../plots/data1_pca.jpg")

eigenvectors, eigenvalues, plot = utils.pca3d("../data2_with_all_numerical_features.csv")
plt.figure()
plt.savefig("../plots/data2_pca.jpg")
plt.show()

step = utils.ratioProgress("../data1.csv")
step.update_layout(
    width=800,
    title_text='Total Explained Variance Progress upon Additional Components for Dataset 1'
)
step.show()

step = utils.ratioProgress("../data2_with_all_numerical_features.csv")
step.update_layout(
    width=800,
    title_text='Total Explained Variance Progress upon Additional Components for Dataset 2'
)
step.show()

tsne = utils.tsneAnalysis("../data2_with_all_numerical_features.csv")
tsne.show()