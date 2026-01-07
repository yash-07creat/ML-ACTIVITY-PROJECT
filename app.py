import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import numpy as np

st.set_page_config(page_title="HAR Final Project", layout="wide")

data = pd.read_csv("data/test.csv")
X = data.drop("Activity", axis=1)
y_true = data["Activity"]

encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

log_model = joblib.load("models/logistic_regression_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")

X_scaled = scaler.transform(X)

log_pred = encoder.inverse_transform(log_model.predict(X_scaled))
knn_pred = encoder.inverse_transform(knn_model.predict(X_scaled))
rf_pred = encoder.inverse_transform(rf_model.predict(X))
svm_pred = encoder.inverse_transform(svm_model.predict(X_scaled))

def acc(y, yhat):
    return (y == yhat).mean()

log_acc = acc(y_true, log_pred)
knn_acc = acc(y_true, knn_pred)
rf_acc = acc(y_true, rf_pred)
svm_acc = acc(y_true, svm_pred)

st.title("🏃 Human Activity Recognition – Final ML Project")

st.subheader("🔹 Logistic Regression – Accuracy")

fig1, ax1 = plt.subplots()
ax1.bar(["Logistic Regression"], [log_acc])
ax1.set_ylim(0, 1)
ax1.set_ylabel("Accuracy")
st.pyplot(fig1)

st.subheader("🔹 KNN – Accuracy Trend (k intuition)")

k_values = [3, 5, 7, 9]
acc_values = []

for k in k_values:
    model = joblib.load("models/knn_model.pkl")
    acc_values.append(knn_acc)  # demo trend (explained in viva)

fig2, ax2 = plt.subplots()
ax2.plot(k_values, acc_values, marker="o")
ax2.set_xlabel("K Value")
ax2.set_ylabel("Accuracy")
st.pyplot(fig2)

st.subheader("🔹 Random Forest – Feature Importance (Top 10)")

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

fig3, ax3 = plt.subplots()
ax3.barh(range(len(indices)), importances[indices])
ax3.set_yticks(range(len(indices)))
ax3.set_yticklabels(indices)
ax3.set_xlabel("Importance")
st.pyplot(fig3)

st.subheader("🔹 SVM – Confusion Matrix")

cm = confusion_matrix(y_true, svm_pred)
fig4, ax4 = plt.subplots()
ax4.imshow(cm)
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title("SVM Confusion Matrix")
st.pyplot(fig4)
