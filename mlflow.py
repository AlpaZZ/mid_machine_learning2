import sys
import os

# Hindari tabrakan nama (name collision) karena script ini bernama mlflow.py
# Kita hapus path direktori lokal dari sys.path agar 'import mlflow' merujuk ke library sistem.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)
while "" in sys.path:
    sys.path.remove("")

# Sekarang kita aman untuk mengimpor library mlflow sistem
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Dataset (Menggunakan nama file yang benar dengan underscore)
file_name = "Preprocessed_Balanced_dataset.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Dataset {file_name} tidak ditemukan! Pastikan sudah mendownloadnya di Tahap 2.")

print(f"Loading dataset {file_name}...")
df = pd.read_csv(file_name)

# 2. Pemisahan Fitur dan Target
kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
X = df.drop(columns=[col for col in kolom_contekan if col in df.columns])
y = df['Label']

# 3. Train-Test Split (Sesuai proporsi di Tahap 4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Data split selesai. Train: {X_train.shape[0]} baris, Test: {X_test.shape[0]} baris.")

# 4. Target Encoding menggunakan LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 5. Inisialisasi Pipeline Terbaik dengan Hyperparameter Statis (Tahap 7 GridSearchCV Winner)
# k=15 untuk SelectKBest, n_estimators=20 untuk RandomForestClassifier
print("Merakit best pipeline...")
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', RandomUnderSampler(random_state=42)),
    ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
    ('model', RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1))
])

# 6. MLflow Tracking (Start Run)
mlflow.set_experiment("IoT_Vulnerability_Detection")

print("Memulai training pipeline terbaik dan pencatatan MLflow...")
with mlflow.start_run(run_name="Best_Pipeline_Model"):
    # Train model
    best_pipeline.fit(X_train, y_train_encoded)
    
    # Predict & Evaluate
    y_pred = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='macro')
    
    # Log Parameters secara statis sesuai requirements
    mlflow.log_param("Feature_Selection_Method", "SelectKBest")
    mlflow.log_param("K_Features", 15)
    mlflow.log_param("Classifier", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 20)
    mlflow.log_param("random_state", 42)
    
    # Log Metrics
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1_Macro", f1)
    
    # Log Model Artifact
    mlflow.sklearn.log_model(best_pipeline, "best_pipeline_model")
    
    print("\n" + "="*50)
    print("🎉 MLflow Tracking Berhasil!")
    print(f"Logged Params: k=15, n_estimators=20")
    print(f"Logged Metrics: Accuracy={acc:.6f}, F1_Macro={f1:.6f}")
    print("="*50 + "\n")

# 7. Simpan Model & Label Encoder ke file pickle untuk Streamlit (Deployment)
print("Menyimpan pipeline_terbaik.pkl dan label_encoder.pkl...")
joblib.dump(best_pipeline, "pipeline_terbaik.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Semua artifacts berhasil disimpan! Siap dideploy. 🚀")
