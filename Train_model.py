import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport

# Đọc dữ liệu
data = pd.read_csv("stroke_classification.csv")

# profile = ProfileReport(data, title="Stroke Report")
# profile.to_file("StrokeReport.html")

# Xử lý missing value
data = data.dropna()

# Xoá cột ID
data.drop(columns=['pat_id'], inplace=True)

# One-hot encoding
data = pd.get_dummies(data, columns=["gender"], drop_first=True)

# Tách X và y
y = data["stroke"]
X = data.drop("stroke", axis=1)

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cân bằng dữ liệu bằng SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Chuẩn hoá dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# profile = ProfileReport(data, title="StrokeReport after Data processing")
# profile.to_file("StrokeReport_after_Data_processing.html") #Cái này mình chạy lại để xem xem sau khi tiền xử lý dữ liệu thì dữ liệu đã biến đổi ra sao

# from lazypredict.Supervised import LazyClassifier
# from sklearn.metrics import recall_score
# # Đánh giá các mô hình phổ biến
# clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=recall_score)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# print(models.columns)

# # Sắp xếp theo Recall để xem mô hình nào tốt nhất
# print(models.sort_values("recall_score", ascending=False))

#                                Accuracy  Balanced Accuracy  ROC AUC  F1 Score  recall_score  Time Taken
# Model
# GaussianNB                         0.23               0.59     0.59      0.30          1.00        0.02
# QuadraticDiscriminantAnalysis      0.06               0.50     0.50      0.02          1.00        0.01
# PassiveAggressiveClassifier        0.75               0.75     0.75      0.82          0.74        0.02
# NearestCentroid                    0.76               0.71     0.71      0.82          0.66        0.01
# BernoulliNB                        0.78               0.71     0.71      0.84          0.64        0.01
# RidgeClassifierCV                  0.78               0.68     0.68      0.84          0.57        0.02

# GaussianNB	                0.23	1.00 ->>> Overfitting cực mạnh, đánh đổi hết precision
# QDA	                        0.06	1.00 ->>> Tệ toàn diện, cũng chỉ đoán 1 class
# PassiveAggressiveClassifier	0.75	0.74 ->>> Cân bằng tốt giữa accuracy và recall, F1 và thời gian chạy cũng rất tốt -> sửu dụng model này!!!

from sklearn.linear_model import PassiveAggressiveClassifier
# Huấn luyện model
model = GridSearchCV(
    estimator=PassiveAggressiveClassifier(),
    param_grid = ({
    "C": [0.001, 0.01, 0.1, 1.0, 10],
    "max_iter": [500, 1000, 2000],
    "tol": [1e-3, 1e-4, 1e-5],
    "loss": ['hinge', 'squared_hinge'],
    "early_stopping": [True, False]
    }),
    scoring="recall",
    cv=4,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
print("Best recall: ",model.best_score_)
print("Best Params:", model.best_params_)
# Đánh giá
print(classification_report(y_test, y_pred))

# Best recall:  0.8756236589643659
# Best Params: {'C': 10, 'early_stopping': True, 'loss': 'squared_hinge', 'max_iter': 1000, 'tol': 0.0001}
#               precision    recall  f1-score   support

#            0       0.96      0.78      0.86       929
#            1       0.08      0.36      0.14        53

#     accuracy                           0.76       982
#    macro avg       0.52      0.57      0.50       982
# weighted avg       0.91      0.76      0.82       982

import joblib

joblib.dump(model.best_estimator_, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")