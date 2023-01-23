import streamlit as st
import pandas as pd

###################### Classification ####################
from sklearn import tree

clf = tree.DecisionTreeClassifier()
st.markdown("<h1 style='text-align: center; '>Decision tree classifier</h1>", unsafe_allow_html=True)

train_percent = st.slider("Percent of data to use for training:", 1, 99, 30)

df = st.session_state.df
variable_list = st.session_state.variable_list
vars_ = st.multiselect('Select the attributes to include:',
    variable_list,
    default=variable_list[:-1])
vars_ = [True if i in vars_ else False for i in variable_list]

df_len = len(df)
cutoff = int(df_len * train_percent / 100)

# shuffle df
df = df.sample(frac=1)

# exclude target from vars_
vars_[-1] = False
clf = clf.fit(df[:cutoff].to_numpy()[:,vars_], df[:cutoff].target)

from sklearn.metrics import classification_report

st.markdown("<h4 style='text-align: center; '>Classification report:</h4>", unsafe_allow_html=True)

report = classification_report(df[cutoff:].target, clf.predict(df[cutoff:].to_numpy()[:,vars_]), output_dict=True)
report = pd.DataFrame(report).transpose()

st.write(report)

from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

st.header("Oversampling with Borderline SMOTE")
st.write("Dataset is already balanced")
st.write(df['target'].value_counts())


X_resampled, y_resampled = BorderlineSMOTE().fit_resample(df, df.target)
c_df = sorted(Counter(y_resampled).items())
# df_resampled = pd.DataFrame.from_dict(Counter(y_resampled).items(), orient='index').reset_index()
df_resampled = pd.DataFrame.from_records(list(dict(c_df).items()))
df_resampled.drop(columns=df_resampled.columns[0], axis=0, inplace=True)
st.write("Equal distribution after Borderline SMOTE")
df_resampled
# X_resampled
# y_resampled


report = classification_report(X_resampled[cutoff:].target, clf.predict(X_resampled[cutoff:].to_numpy()[:,vars_]), output_dict=True)
report = pd.DataFrame(report).transpose()
st.write(report)