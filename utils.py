import os
import json
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model

# AUTO VALIDATION ARTIFACT LOADER
def load_artifacts(artifact_dir):
    # 1. Validasi folder
    if not os.path.exists(artifact_dir):
        st.error(f"Folder artefak tidak ditemukan:\n{artifact_dir}")
        st.stop()

    if not os.path.isdir(artifact_dir):
        st.error(f"Path bukan folder:\n{artifact_dir}")
        st.stop()

    # 2. File wajib
    required_files = [
        "model.keras",
        "scaler_target.pkl",
        "scaler_features.pkl",
        "model_metadata.json"
    ]

    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(artifact_dir, f)):
            missing.append(f)

    if missing:
        st.error("File artefak berikut tidak ditemukan:")
        for f in missing:
            st.write(f"• {f}")
        st.stop()

    # 3. Load model
    try:
        model = load_model(
            os.path.join(artifact_dir, "model.keras")
        )
    except Exception as e:
        st.error("Gagal memuat model")
        st.exception(e)
        st.stop()

    # 4. Load scaler
    try:
        with open(os.path.join(artifact_dir, "scaler_target.pkl"), "rb") as f:
            scaler_target = pickle.load(f)

        with open(os.path.join(artifact_dir, "scaler_features.pkl"), "rb") as f:
            scaler_features = pickle.load(f)
    except Exception as e:
        st.error("Gagal memuat scaler")
        st.exception(e)
        st.stop()

    # 5. Load metadata
    try:
        with open(os.path.join(artifact_dir, "model_metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception as e:
        st.error("Gagal memuat metadata")
        st.exception(e)
        st.stop()

    # 6. Validasi metadata
    for key in ["target_column", "feature_columns", "timestep"]:
        if key not in metadata:
            st.error(f"Metadata tidak valid, key '{key}' tidak ditemukan")
            st.stop()

    # 7. Load history (opsional)
    history = None
    history_path = os.path.join(artifact_dir, "training_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = None

    st.success("Model dan artefak berhasil dimuat")

    return model, scaler_target, scaler_features, metadata, history

# WINDOWING FUNCTION
def create_window(data, time_step):
    X = []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, :])
    return np.array(X)
