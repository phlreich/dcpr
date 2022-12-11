# Basic imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
# sklearn imports
from sklearn.decomposition import PCA  # Principal Component Analysis

from sklearn.cluster import KMeans

# plotly imports
import warnings
from pages.Data_Imputation import deletion_df, mean_df, median_df


warnings.simplefilter('ignore')
sns.set()

categorical = ['sex', 'fbs', 'cp', 'restecg', 'ca', 'thal', 'exang', 'target']
st.sidebar.markdown("### Test Sidebar")

original_df = pd.read_csv("data/heart.csv")
datasets = {
    'Original Dataset': original_df,
    'Median Imputation': median_df,
    'Mean Imputation': mean_df,
    'Listwise Deletion': deletion_df
}

dataset = st.sidebar.radio(
    "Select the dataset",
    datasets.keys()
)

st.title("Clustering")
st.subheader(f"Dataset: {dataset}")
df = datasets[dataset]
df

# oldpeak to int
df['oldpeak'] = df['oldpeak'].astype(int)
# categorical to object
df['sex'] = df['sex'].astype(object)
df['cp'] = df['cp'].astype(object)
df['fbs'] = df['fbs'].astype(object)
df['restecg'] = df['restecg'].astype(object)
df['exang'] = df['exang'].astype(object)
df['slope'] = df['slope'].astype(object)
df['ca'] = df['ca'].astype(object)
df['thal'] = df['thal'].astype(object)
df['target'] = df['target'].astype(object)

"""
The first step is to perform one-hot encoding on the categorical data\n
in order to compute the euclidean distance with k-means.
"""
df = pd.get_dummies(df, prefix=categorical, columns=categorical)

st.subheader("One-Hot Encoded")
df

"Then the numerical data is normalized using Min-Max Scaler"
from sklearn.preprocessing import MinMaxScaler

df_norm = df.copy()
scaler = MinMaxScaler()
# scaler.fit(data)
df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
    df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

st.subheader("Normalized with Min-Max Scaler")
df_norm

st.subheader("Perform dimensionality reduction with PCA")
"""
We will use these principal components to help us visualize our clusters in 1-D, 2-D, and 3-D space,\n
since we cannot easily visualize the data we have in higher dimensions.\n
For example, we can use two principal components to visualize the clusters in 2-D space,\n
or three principal components to visualize the clusters in 3-D space.
"""

# Perform dimensionality reduction with PCA
# PCA with one principal component
pca_1d = PCA(n_components=1)

# PCA with two principal components
pca_2d = PCA(n_components=2)

# This DataFrame contains the two principal components that will be used
# for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(df_norm))

# "PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
# "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

df_norm = pd.concat([df_norm, PCs_2d], axis=1, join='inner')

df_norm

st.header("Clustering")

st.subheader("K-Means")
"""First step is to find the optimal number of clusters with te help of the elbow method\n
Sources:\n
https://www.kaggle.com/code/minc33/visualizing-high-dimensional-clusters/notebook\n
https://towardsdatascience.com/clustering-on-numerical-and-categorical-features-6e0ebcf1cbad\n
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/\n
"""

fig_elbow, ax = plt.subplots()
distortions = []
Y = df_norm
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(Y)
    distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(8, 5))

ax.plot(K, distortions, 'bx-')
# ax.xlabel('k')
# ax.ylabel('Distortion')
# ax.title('The Elbow Method showing the optimal k')
st.pyplot(fig_elbow)

# st.line_chart(data=distortions)

from sklearn.metrics import silhouette_score

# Initialize our model
kmeans = KMeans(n_clusters=3)
# Fit our model
clusters = kmeans.fit(df_norm)
# Find which cluster each data-point belongs to
label = kmeans.predict(df_norm)
# Add the cluster vector to our DataFrame, X
df_norm["Cluster"] = clusters.labels_

"""
The Silhouette score is a metric used to calculate the goodness of a clustering technique.\n
Its value ranges from -1 to 1\n
1: Clusters are well apart from each other and clearly distinguished.\n
0: Clusters are indifferent, or we can say that the distance between clusters is not significant.\n
-1: Clusters are assigned in the wrong way.\n\n

Source:\n
https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c#:~:text=Silhouette%20Coefficient%20or%20silhouette%20score%20is%20a%20metric%20used%20to,each%20other%20and%20clearly%20distinguished.
"""
st.subheader(f"Silhouette Score(n=2): {silhouette_score(df_norm, label)}")

fig = plt.figure(figsize=(10, 4))
p = sns.scatterplot(data=df_norm, x="PC1_2d", y="PC2_2d", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')

st.pyplot(fig)
# ----------- # ----------- # ----------- # ----------- # -----------

st.subheader("DBSCAN")
"""
Sources:\n
https://www.reneshbedre.com/blog/dbscan-python.html\n
https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
"""

df_dbscan = df_norm.copy()
df_dbscan = df_norm[['PC1_2d', 'PC2_2d']]

from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df_dbscan)

distances, indices = nbrs.kneighbors(df_dbscan)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

"""
Find the optimal value for epsilon
Source: https://www.reneshbedre.com/blog/dbscan-python.html
"""

fig_dbscan = plt.figure()
plt.plot(distances)
st.pyplot(fig_dbscan)

from sklearn.cluster import DBSCAN

# Configuring the parameters of the clustering algorithm
# eps: maximum distance between two points
# min_samples: the larger the data set, the larger the no. of minimum points should be
dbscan_cluster = DBSCAN(eps=0.3, min_samples=30)

# Fitting the clustering algorithm
dbscan_cluster.fit(df_dbscan)

# Adding the results to a new column in the dataframe
df_dbscan["cluster"] = dbscan_cluster.labels_

from collections import Counter

cluster_counter = Counter(dbscan_cluster.labels_)
"Number of nodes in clusters"
st.text(cluster_counter)

fig_dbscan = plt.figure()
p = sns.scatterplot(data=df_dbscan, x="PC1_2d", y="PC2_2d", hue=dbscan_cluster.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
st.pyplot(fig_dbscan)
