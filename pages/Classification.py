import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import tree
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC

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

training_df = df[:cutoff]
test_df = df[cutoff:]

exclude_percent = st.slider("Percent of data where target=1 to exclude from training data:", 0, 99, 0)

df_len = len(training_df[training_df.target == 1])
exclude_cutoff = df_len - int(df_len * exclude_percent / 100)
training_df = pd.concat([training_df[training_df.target == 0], training_df[training_df.target == 1][:exclude_cutoff]])

# show data balance
st.write("Data balance:")
st.write(training_df.target.value_counts())

# exclude target from vars_
vars_[-1] = False

st.header("Unbalanced training data results:")

# train with unaltered data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_df.to_numpy()[:,vars_], training_df.target)

# show classification report for unaltered training data
report = classification_report(test_df.target, clf.predict(test_df.to_numpy()[:,vars_]), output_dict=True)
report = pd.DataFrame(report).transpose()
st.write(report)

st.header("Oversampling with Borderline SMOTE")

X_resampled, y_resampled = BorderlineSMOTE().fit_resample(training_df, training_df.target)

# train with oversampled data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_resampled.to_numpy()[:,vars_], X_resampled.target)

# show classification report for oversampled training data
report = classification_report(test_df.target, clf.predict(test_df.to_numpy()[:,vars_]), output_dict=True)
report = pd.DataFrame(report).transpose()
st.write(report)


st.header("SMOTENC results")
categorical_mask = [False, True, True, False, False, True, True, False, True, False, True, True, True, True]
X_resampled, y_resampled = SMOTENC(categorical_mask).fit_resample(training_df, training_df.target)

# train with oversampled data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_resampled.to_numpy()[:,vars_], X_resampled.target)

# show classification report for oversampled training data
report = classification_report(test_df.target, clf.predict(test_df.to_numpy()[:,vars_]), output_dict=True)
report = pd.DataFrame(report).transpose()
st.write(report)