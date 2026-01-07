import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop("Activity", axis=1)
y_train = train["Activity"]
X_test = test.drop("Activity", axis=1)
y_test = test["Activity"]

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)
joblib.dump(encoder, "models/label_encoder.pkl")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

log_model = LogisticRegression(max_iter=2000)
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
svm_model = SVC(kernel="rbf")

log_model.fit(X_train_scaled, y_train_enc)
knn_model.fit(X_train_scaled, y_train_enc)
rf_model.fit(X_train, y_train_enc)
svm_model.fit(X_train_scaled, y_train_enc)

joblib.dump(log_model, "models/logistic_regression_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")

print("✅ All models trained and saved")
