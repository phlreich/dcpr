import pandas as pd
import streamlit as st
from random import randint
import numpy as np


@st.cache
def load_data(data_path: str):
    data = pd.read_csv(data_path)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


def delete_with_probability(val):
    if randint(0, 100) <= 5:
        val = np.NAN
    return val


def delete_random_values(df: pd.DataFrame):
    """Delete values in given columns randomly with probability 0.05"""
    cols = ['slope', 'exang', 'restecg', 'fbs', 'cp'] # to delete values in
    for col in cols:
        df[col] = df[col].apply(delete_with_probability)
    return df

