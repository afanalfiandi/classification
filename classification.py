import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from mahotas.features import haralick


# =========[ FITUR EKSTRAKSI GLCM ]=========
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = haralick(gray).mean(axis=0)  # 13 fitur Haralick (rata-rata arah)
    return features


# =========[ LOAD DATASET FOLDER KE FITUR DAN LABEL ]=========
def load_dataset(base_path):
    X, y = [], []
    for label_folder in os.listdir(base_path):
        label_folder_path = os.path.join(base_path, label_folder)
        if not os.path.isdir(label_folder_path):
            continue
        for file in os.listdir(label_folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(label_folder_path, file)
                try:
                    features = extract_features(img_path)
                    X.append(features)
                    y.append(label_folder)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)


# =========[ PATH DATASET ]=========
train_path = "dataset/training"
valid_path = "dataset/valid"
test_path = "dataset/testing"

# =========[ LOAD DATA ]=========
X_train, y_train = load_dataset(train_path)
X_valid, y_valid = load_dataset(valid_path)
X_test, y_test = load_dataset(test_path)

# =========[ ENKODE LABEL ]=========
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_valid_enc = le.transform(y_valid)
y_test_enc = le.transform(y_test)

# =========[ TRAIN MODEL SVM ]=========
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train_enc)

# =========[ PREDIKSI TESTING DATA ]=========
y_pred = model.predict(X_test)

# =========[ BLOK EVALUASI KLASIFIKASI ]=========

# 1. CLASSIFICATION REPORT
print("\n📄 Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# 2. CONFUSION MATRIX
cm = confusion_matrix(y_test_enc, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# 3. PERHITUNGAN TP, FP, TN, FN PER KELAS
num_classes = len(le.classes_)
for i, label in enumerate(le.classes_):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    print(f"\n📌 Class: {label}")
    print(f"   TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

# 4. PRECISION, RECALL, F1-SCORE, SUPPORT PER KELAS
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_enc, y_pred, zero_division=0
)

print("\n📈 Metrics per Class:")
for idx, label in enumerate(le.classes_):
    print(f"- {label}")
    print(f"  Precision: {precision[idx]:.2f}")
    print(f"  Recall:    {recall[idx]:.2f}")
    print(f"  F1-score:  {f1[idx]:.2f}")
    print(f"  Support:   {support[idx]}")

# =========[ END BLOK ]=========
