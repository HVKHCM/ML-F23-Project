import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scikitplot.metrics import plot_roc
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d import proj3d

from sklearn.manifold import TSNE

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


def get_data(data_csv_path:str): #https://pandas.pydata.org/pandas-docs/stable/reference/frame.html 
    df = pd.read_csv(data_csv_path)
    df.columns=["x"+str(i) for i in range(len(df.columns))]
    df.rename(columns = {list(df)[len(df.columns)-1]:'label'}, inplace=True)
    labels_list = []
    labels_df = df["label"]

    for label in labels_df:
        labels_list.append(label)

    labels = np.array(labels_list)

    df_without_labels = df.drop("label", axis=1)

    return df_without_labels, labels

def pca3d(datapath):
    dataset_X, dataset_y = get_data(datapath)
    x = StandardScaler().fit_transform(dataset_X)

    pca = PCA(n_components=3)
    pcaNorm= pca.fit_transform(x)

    total_var = pca.explained_variance_ratio_.sum() * 100

    pcaFrame = pd.DataFrame(data = pcaNorm
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)
    ax.set_zlabel('C3', fontsize=20)
    plt.title("Principal Component Analysis of Dataset 1 \n Total Explained Variance : {:.2f}%".format(total_var),fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = dataset_y == target
        ax.scatter3D(pcaFrame.loc[indicesToKeep, 'principal component 1'], 
                pcaFrame.loc[indicesToKeep, 'principal component 2'], 
              pcaFrame.loc[indicesToKeep, 'principal component 3'], 
                c = color, s = 10)
        
    for vec in pca.components_:
        a = Arrow3D([0, vec[0]*45], [0, vec[1]*45], [0, vec[2]*45], mutation_scale=10, 
                lw=3, arrowstyle="-|>", color="blue", linestyle='dashed')
        ax.add_artist(a)


    plt.legend(targets,prop={'size': 15})

    return pca.components_, pca.explained_variance_, ax

def ratioProgress(datapath):
    dataset_X, dataset_y = get_data(datapath)
    x = StandardScaler().fit_transform(dataset_X)
    normalized_feature = ['Nfeature'+str(i) for i in range(x.shape[1])]
    normFrame= pd.DataFrame(x,columns=normalized_feature)

    pca = PCA(n_components=x.shape[1])
    pcaNorm= pca.fit_transform(x)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)


    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )

    return fig

def tsneAnalysis(datapath):
    tsne = TSNE(n_components=3, random_state=42)
    dataset_X, dataset_y = get_data(datapath)
    x = StandardScaler().fit_transform(dataset_X)
    X_tsne = tsne.fit_transform(x)
    tsne.kl_divergence_

    fig = px.scatter_3d(x=X_tsne[:, 0], y=X_tsne[:,1], z=X_tsne[:,2],color=dataset_y, color_continuous_scale="earth")
    fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    scene=dict(
        xaxis_title='First Component',
        yaxis_title='Second Component',
        zaxis_title='Third Component',
    ),)
    
    return fig

