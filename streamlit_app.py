import streamlit as st 


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


st.subheader('Dataset sample:')
# st.dataframe(df.sample(10))  # Same as st.write(df)
df

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


###### PACMAP ################

v1 = st.checkbox('Exclude from PaCMAP: 1. age')
v2 = st.checkbox('Exclude from PaCMAP: 2. sex')
v3 = st.checkbox('Exclude from PaCMAP: 3. cp: chest pain type (4 values)')
v4 = st.checkbox('Exclude from PaCMAP: 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)')
v5 = st.checkbox('Exclude from PaCMAP: 5. chol: serum cholestoral in mg/dl')
v6 = st.checkbox('Exclude from PaCMAP: 6. fbs: fasting blood sugar > 120 mg/dl')
v7 = st.checkbox('Exclude from PaCMAP: 7. restecg: resting electrocardiographic results (values 0,1,2)')
v8 = st.checkbox('Exclude from PaCMAP: 8. thalach: maximum heart rate achieved')
v9 = st.checkbox('Exclude from PaCMAP: 9. exang: exercise induced angina')
v10 = st.checkbox('Exclude from PaCMAP: 10. oldpeak: ST depression induced by exercise relative to rest')
v11 = st.checkbox('Exclude from PaCMAP: 11. slope: the slope of the peak exercise ST segment')
v12 = st.checkbox('Exclude from PaCMAP: 12. ca: number of major vessels (0-3) colored by flourosopy')
v13 = st.checkbox('Exclude from PaCMAP: 13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
v14 = st.checkbox('Exclude from PaCMAP: 14. target: 1 = disease; 0 = no disease')

vars = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]
vars = [not i for i in vars]

#df.columns.to_numpy().reshape(1,-1)[:,vars][:5]
#df.to_numpy()[:,vars][:2]

select_sex = st.slider('0 for female, 1 for male, 2 for both:', 0, 1, 2)

st.write('PaCMAP dim reduction. Dark color indicates healthy, light color indicates disease:')

if select_sex != 2:
    df = df[df["sex"] == select_sex]

import pacmap as pm

def pacmap(df, target=True):

    # create pacmap object
    pac = pm.PaCMAP()

    # fit pacmap

    reduced = pac.fit_transform(df.to_numpy()[:,vars], init="pca")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=df.target)#, cmap="Spectral")



st.pyplot(pacmap(df))


############# t-sne ###################

st.write("t-SNE:")

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, 
    learning_rate='auto', init='random', perplexity=3).fit_transform(df.to_numpy())

tsnefig, tsneax = plt.subplots(1, 1, figsize=(6, 6))
tsneax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df.target)#, cmap="Spectral")

st.pyplot(tsnefig)
