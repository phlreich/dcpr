# Basic imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

sns.set()

# plotly imports

import warnings

warnings.simplefilter('ignore')

from sklearn.neighbors import KNeighborsClassifier


st.title("KNN & SVM")

df = pd.read_csv("data/heart.csv")

df_norm = df.copy()
scaler = MinMaxScaler()

df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
    df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

st.header("1. Normalize the data using Min-max scaler")
df_norm

st.header("2. Transform categorical to normal features using One-hot Encoding")
# We use pandas's 'get_dummies()' method for hot-encoding
df_norm = pd.get_dummies(df_norm, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
df_norm

st.header("3. Checking for Imbalances")
"Checking for imbalanced data based on outcome."
st.write(df_norm['target'].value_counts())

st.header("4. Creating Imbalance")

"delete the first 400 rows where target = 1"
df_norm = df_norm.sort_values(by='target')
df_norm.head(50)
df_norm = df_norm.iloc[400:]

df_norm
"Imbalanced Data"
st.write(df_norm['target'].value_counts())

st.header("5. Splitting train & test data")

"Test data size: 25%"

X = df_norm.iloc[:, df_norm.columns != 'target'].values
y = df_norm['target'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

st.header("6. KNN")
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
st.pyplot(plt)
st.write("Minimum error:-", min(error), "at K =", error.index(min(error)) + 1)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

from sklearn.metrics import classification_report

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.header("7. SVM")
# Creating SVM model.

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(clf, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.header("8. Oversampling")
"https://medium.com/grabngoinfo/four-oversampling-and-under-sampling-methods-for-imbalanced-classification-using-python-7304aedf9037"

# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

st.header("8.1 ROS")
"One way of oversampling is to generate new samples for the minority class by sampling with replacement. " \
"The RandomOverSampler from the imblearn library provides such functionality."

# Randomly over sample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
# Check the number of records after over sampling
st.write(sorted(Counter(y_train_ros).items()))

st.subheader("KNN")
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_ros, y_train_ros)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
st.write("Minimum error:-", min(error), "at K =", error.index(min(error)) + 1)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train_ros, y_train_ros)
y_pred = classifier.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.subheader("SVM")
# Creating SVM model.

clf = svm.SVC(kernel='rbf')
clf.fit(X_train_ros, y_train_ros)
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.header("8.2 SMOTE")
"Instead of randomly oversampling with replacement, SMOTE takes each minority sample and " \
"introduces synthetic data points connecting the minority sample and its nearest neighbors. " \
"Neighbors from the k nearest neighbors are chosen randomly."
# Randomly over sample the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Check the number of records after over sampling
st.write(sorted(Counter(y_train_smote).items()))

st.subheader("KNN")
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_smote, y_train_smote)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
st.write("Minimum error:-", min(error), "at K =", error.index(min(error)) + 1)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train_smote, y_train_smote)
y_pred = classifier.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.subheader("SVM")
# Creating SVM model.

clf = svm.SVC(kernel='rbf')
clf.fit(X_train_smote, y_train_smote)
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.header("9. Random Undersampling")
st.subheader("9.1 RUS")
"Random under-sampling randomly picks data points from the majority class. " \
"After the sampling, the majority class should have the same number of data points as the minority class."

# Randomly under sample the majority class
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
# Check the number of records after under sampling
st.write(sorted(Counter(y_train_rus).items()))

st.subheader("KNN")
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_rus, y_train_rus)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
st.write("Minimum error:-", min(error), "at K =", error.index(min(error)) + 1)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train_rus, y_train_rus)
y_pred = classifier.predict(X_test)
disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.subheader("SVM")
# Creating SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train_rus, y_train_rus)
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.header("9.2 NearMiss-3")
"“NearMiss-3 is a 2-steps algorithm. First, for each negative sample, their M nearest-neighbors will be kept. " \
"Then, the positive samples selected are the one for which the average distance to the " \
"N nearest-neighbors is the largest.”"
# Under sample the majority class
nearmiss = NearMiss(version=3)
X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(X_train, y_train)
# Check the number of records after over sampling
st.write(sorted(Counter(y_train_nearmiss).items()))

st.subheader("KNN")
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_nearmiss, y_train_nearmiss)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
st.write("Minimum error:-", min(error), "at K =", error.index(min(error)) + 1)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train_nearmiss, y_train_nearmiss)
y_pred = classifier.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.subheader("SVM")
# Creating SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train_nearmiss, y_train_nearmiss)
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap='Reds')
st.pyplot()

st.subheader("Classification Report")
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
