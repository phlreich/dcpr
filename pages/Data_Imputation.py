import streamlit as st
import pandas as pd
from random import randint
import numpy as np
from utils.utils import *

imputation_feats = ['slope', 'exang', 'restecg', 'fbs', 'cp']
categorical_feats = ['cp', 'restecg', 'slope', 'thal']
numerical_feats = ['oldpeak', 'trestbps', 'chol', 'thalach']
binary_feats = ['sex', 'fbs', 'exang']


def impute_mean(df: pd.DataFrame):
    """Replace NAN with mean of column"""
    ret_df = df.copy()
    for feat in imputation_feats:
        mean = ret_df[feat].mean()
        ret_df[feat] = ret_df[feat].fillna(mean)

    return ret_df


def impute_median(df: pd.DataFrame):
    """Replace NAN with median of column"""
    ret_df = df.copy()
    for feat in imputation_feats:
        mean = ret_df[feat].median()
        ret_df[feat] = ret_df[feat].fillna(mean)

    return ret_df


def impute_listwise_deletion(df: pd.DataFrame):
    """Remove a row where at least one value is NAN"""
    return df.dropna()


st.title("Data Imputation")

heart_df = pd.read_csv("data/heart_del.csv")
mis_features = ['thal', 'ca', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'sex']
#heart_df[mis_features] = heart_df[mis_features].astype(object)


st.subheader("Values Deleted")
heart_df

st.subheader("Mean Imputation")
mean_df = impute_mean(heart_df)
mean_df

st.subheader("Median Imputation")
median_df = impute_median(heart_df)
median_df

st.subheader("Listwise Deletion 'Imputation'")
deletion_df = impute_listwise_deletion(heart_df)
deletion_df

sum_deletion_df = len(deletion_df)
st.text(f"Number of rows: {sum_deletion_df}")


