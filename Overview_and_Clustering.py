from pages.Data_Imputation import median_df, mean_df, deletion_df
import streamlit as st 

import pacmap as pm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import statistics
import sys
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

imputation_feats = ['slope', 'exang', 'restecg', 'fbs', 'cp']

def delete_with_probability(val):
    if randint(0, 100) <= 5:
        val = np.NAN
    return val


def delete_random_values(df: pd.DataFrame):
    """Delete values in given columns randomly with probability 0.05"""
    cols = ['slope', 'exang', 'restecg', 'fbs', 'cp']
    for col in cols:
        df[col] = df[col].apply(delete_with_probability)
    return df
    
def impute_mean(df: pd.DataFrame):
    """Replace NAN with mean of column"""
    for feat in imputation_feats:
        mean = df[feat].mean()
        df[feat] = df[feat].fillna(mean)

    return df


def impute_median(df: pd.DataFrame):
    """Replace NAN with median of column"""
    for feat in imputation_feats:
        mean = df[feat].median()
        df[feat] = df[feat].fillna(mean)

    return df


def impute_listwise_deletion(df: pd.DataFrame):
    """Remove a row where at least one value is NAN"""
    return df.dropna()

def get_silhouette_coefficient(X, kmeans, metric='euclidean'):
    """Silhouette Coefficient for kmeans clustering"""
    labels = kmeans.labels_
    st.write("Silhouette Coefficient (The higher the score the better the clustering [0,1]; -1 invalid cluster):")
    if max(labels) < 1:
        return st.write("Choose at least k=2 to get a valid coefficient!")
    else:
        coefficient = metrics.silhouette_score(X, labels, metric=metric)
        return st.write(coefficient)

def get_davies_bouldin_score(X, kmeans):
    labels = kmeans.labels_
    st.write("Davies Bouldin Score (The lower the better):")
    if max(labels) < 1:
        return st.write("Choose at least k=2 to get a valid score!")
    else:
        score = metrics.davies_bouldin_score(X, labels)
        return st.write(score)

df = pd.read_csv("data/heart.csv")


st.title('Heart Disease Dataset')
st.text("""This data set dates from 1988 and consists of four databases:\n
Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes,\n
including the predicted attribute, but all published experiments refer to using\n
a subset of 14 of them with 1025 entries. The "target" field refers to the presence\n
of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.""")
st.write('Attributes:')
st.write('1. age')
st.write('2. sex')
st.write('3. cp: chest pain type (4 values)')
st.write('4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)')
st.write('5. chol: serum cholestoral in mg/dl')
st.write('6. fbs: fasting blood sugar > 120 mg/dl')
st.write('7. restecg: resting electrocardiographic results (values 0,1,2)')
st.write('8. thalach: maximum heart rate achieved')
st.write('9. exang: exercise induced angina')
st.write('10. oldpeak: ST depression induced by exercise relative to rest')
st.write('11. slope: the slope of the peak exercise ST segment')
st.write('12. ca: number of major vessels (0-3) colored by flourosopy')
st.write('13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
st.write('14. target: 1 = disease; 0 = no disease')

original_df = pd.read_csv("data/heart.csv")
datasets = {
    'Original Dataset': original_df,
    'Median Imputation': median_df,
    'Mean Imputation': mean_df,
    #'Listwise Deletion': deletion_df
}

dataset = st.sidebar.radio(
    "Select the dataset",
    datasets.keys()
)

st.title("Clustering")
st.subheader(f"Dataset: {dataset}")
df = datasets[dataset]
df

# write df to streamlit session state
st.session_state.df = df

######
# Fixing the Data Types
mis_features=['thal','ca','slope','exang','restecg','fbs','cp','sex']
df[mis_features] = df[mis_features].astype(object)

#Split numerical-categorical Features
numerical_col = df.select_dtypes(exclude=np.object_)
categorical_col = df.select_dtypes(exclude=np.number)
######

st.subheader('Dataset visualization:')
# plot of crosstab histograms
st.text('Relationship between sex and presence of a heart disease:')
fig1 = plt.figure(figsize = (6,6))
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(6, 6))
plt.xlabel("0 = No Disease \n 1 = Disease")
plt.xticks(rotation=360)
plt.ylabel("Frequency")
plt.legend(["Female", "Male"])
fig1 = plt.show()
#fig1 = plt.figure()
st.pyplot(fig1)
##################

# correlation matrix plot
st.text('Correlation matrix of the attributes:')
correlation_matrix = df.corr()
fig2 = plt.figure(figsize = (10,6))
sns.heatmap(correlation_matrix, annot = True, cmap="YlGnBu")
st.pyplot(fig2)           
##################

# attribute and target frequency plot
st.text('Relationship between each attribute and the presence of a heart disease:')
cat_col=categorical_col.columns
fig3 = plt.figure(figsize=(12,12))
for index in range(len(cat_col)):
    if cat_col[index] != 'target':
        plt.subplot(4,2,index + 1)
        sns.countplot(data = categorical_col,x=cat_col[index],hue=df['target'], palette ="viridis")
        plt.xlabel(cat_col[index].upper(),fontsize=12)
        plt.ylabel("count", fontsize=12)
        plt.subplots_adjust(wspace = 0.3, hspace= 0.3)
st.pyplot(fig3)


############################################################################

variable_list = ["1. age",
"2. sex",
"3. cp: chest pain type (4 values)",
"4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)",
"5. chol: serum cholestoral in mg/dl",
"6. fbs: fasting blood sugar > 120 mg/dl",
"7. restecg: resting electrocardiographic results (values 0,1,2)",
"8. thalach: maximum heart rate achieved",
"9. exang: exercise induced angina",
"10. oldpeak: ST depression induced by exercise relative to rest",
"11. slope: the slope of the peak exercise ST segment",
"12. ca: number of major vessels (0-3) colored by flourosopy",
"13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect",
"14. target: 1 = disease; 0 = no disease",]

st.session_state.variable_list = variable_list

vars_ = st.multiselect('Select the attributes to include in the clustering:',
    variable_list,
    default=variable_list)

#st.write('You selected:', vars_)

vars_ = [True if i in vars_ else False for i in variable_list]

select_sex = st.select_slider("",options=["both sexes included", "only male", "only female",])

if select_sex != "both sexes included":
    if select_sex == "only female": 
        select_sex = 0
    else:
        select_sex = 1
    df = df[df["sex"] == select_sex]


categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target']
categorical = [i for i in categorical if i in df.columns]

#one-hot encoding is not necessary for KMeans clustering (it is for other clustering methods) 
#one_hot_encode = st.checkbox('One-hot encode categorical features', value=True)
#if one_hot_encode:
#    pd.get_dummies(df, columns=categorical)

min_max_normalize = st.checkbox('Min-max normalize numerical features', value=True)
if min_max_normalize:
    columns_to_normalize = [i for i in df.columns if i not in categorical]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])




############# Task2: clusters ###################

k = st.slider('Number of k-means clusters:', 1, 10, 2)

kmeans = KMeans(n_clusters=k, random_state=0).fit(df.to_numpy()[:,vars_])


# DBSCAN
from sklearn.cluster import DBSCAN 
eps = st.slider('DBSCAN eps:', 0.0, 2.0, 0.5)
min_samples = st.slider('DBSCAN min_samples:', 1, 10, 1)
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df.to_numpy()[:,vars_])


############## Choose coloring scheme

coloring_name = st.selectbox('Select data coloring scheme:', ["target", "sex", "KMEANS cluster", "DBSCAN cluster"])

if coloring_name == "KMEANS cluster":
    coloring = kmeans.labels_
elif coloring_name == "target":
    coloring = df.target
elif coloring_name == "DBSCAN cluster":
    coloring = dbscan.labels_
else:
    coloring = df.sex


st.markdown("<h1 style='text-align: center; '>Scores in high dimensions</h1>", unsafe_allow_html=True)
non_reduced = df.to_numpy()[:,vars_]
#st.write("K-means scores:")
st.markdown("<h4 style='text-align: center; '>K-means scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(non_reduced, kmeans)
get_davies_bouldin_score(non_reduced, kmeans)

#st.write("DBSCAN scores:")
st.markdown("<h4 style='text-align: center; '>DBSCAN scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(non_reduced, dbscan)
get_davies_bouldin_score(non_reduced, dbscan)
############# t-sne ############################

X_embedded = TSNE(n_components=2, learning_rate="auto", init='pca', perplexity=20).fit_transform(df.to_numpy())
tsne_plot = plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=coloring)
plt.title("t-SNE, coloring:" + coloring_name)
st.pyplot(tsne_plot)

#st.write("K-means scores:")
st.markdown("<h4 style='text-align: center; '>K-means scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(X_embedded, kmeans)
get_davies_bouldin_score(X_embedded, kmeans)

#st.write("DBSCAN scores:")
st.markdown("<h4 style='text-align: center; '>DBSCAN scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(X_embedded, dbscan)
get_davies_bouldin_score(X_embedded, dbscan)

###################### PACMAP ########################

# create pacmap object
pac = pm.PaCMAP()
# fit pacmap
reduced = pac.fit_transform(df.to_numpy(), init="pca")
# plot
figpm, axpm = plt.subplots(1, 1, figsize=(6, 6))
axpm.scatter(reduced[:, 0], reduced[:, 1], c=coloring)
axpm.set_title("PaCMAP, coloring: " + coloring_name)
st.pyplot(figpm)

#st.write("K-means scores:")
st.markdown("<h4 style='text-align: center; '>K-means scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(reduced, kmeans)
get_davies_bouldin_score(reduced, kmeans)

#st.write("DBSCAN scores:")
st.markdown("<h4 style='text-align: center; '>DBSCAN scores:</h4>", unsafe_allow_html=True)
get_silhouette_coefficient(reduced, dbscan)
get_davies_bouldin_score(reduced, dbscan)
