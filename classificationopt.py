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
    features = haralick(gray).mean(axis=0)
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
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

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

# =========[ PREDIKSI TESTING DATA & SIMPAN HASIL VISUAL + CSV ]=========
results = []
print("\n🖼️ Memproses dan menyimpan hasil prediksi gambar di dataset/testing:")
for label_folder in os.listdir(test_path):
    label_folder_path = os.path.join(test_path, label_folder)
    if not os.path.isdir(label_folder_path):
        continue
    for file in os.listdir(label_folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(label_folder_path, file)
            try:
                image = cv2.imread(img_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features = haralick(gray).mean(axis=0).reshape(1, -1)

                pred = model.predict(features)[0]
                pred_label = le.inverse_transform([pred])[0]
                true_label = label_folder
                status = "Benar" if pred_label == true_label else "Salah"

                # Simpan gambar dengan anotasi prediksi
                annotated_image = image.copy()
                cv2.putText(
                    annotated_image,
                    f"Prediksi: {pred_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if status == "Benar" else (0, 0, 255),
                    2,
                )
                output_img_path = os.path.join(output_dir, f"pred_{file}")
                cv2.imwrite(output_img_path, annotated_image)

                # Simpan ke list
                results.append(
                    {
                        "filename": file,
                        "label_asli": true_label,
                        "prediksi_model": pred_label,
                        "status": status,
                    }
                )
                print(f"✅ {file} => {pred_label} ({status})")

            except Exception as e:
                print(f"❌ Gagal memproses {file}: {e}")

# Simpan ke CSV
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "hasil_prediksi.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Hasil prediksi disimpan di: {csv_path}")
