# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import plotly.express as px
from datetime import datetime

# === CONFIGURATION ===
st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🥔", layout="wide")
USER_DB = "users.json"
HISTORY_DB = "history.json"
MODEL_PATH = "potato_model_final.h5"

# === DOWNLOAD MODEL FROM GDRIVE ===
if not os.path.exists(MODEL_PATH):
    import gdown
    file_id = "1-0MzkmYfRt0F--10VOAVrHDjd5PvcY-z"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# === STYLING ===
st.markdown("""
    <style>
        .title {text-align: center; font-size: 3em; font-weight: bold; color: #4a4a4a;}
        .subtext {text-align: center; font-size: 1.2em; color: #6e6e6e;}
        .footer {text-align: center; font-size: 1em; color: #999; margin-top: 50px;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            margin-top: 10px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# === UTILITIES ===
def load_json(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def authenticate(username, password):
    users = load_json(USER_DB)
    return username in users and users[username] == password

def register_user(username, password):
    users = load_json(USER_DB)
    if username in users:
        return False
    users[username] = password
    save_json(users, USER_DB)
    return True

def save_prediction(username, prediction):
    history = load_json(HISTORY_DB)
    if username not in history:
        history[username] = []
    history[username].append(prediction)
    save_json(history, HISTORY_DB)

def convert_np(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()
    return obj

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# === DISEASE INFO ===
def disease_info(predicted_class):
    if predicted_class == "Early_Blight":
        st.info("""
        **Early Blight**
        - **Symptoms**: Dark spots on older leaves, concentric rings, yellowing.
        - **Prevention**: Use disease-free seeds, proper crop rotation.
        - **Cure**: Apply fungicides like chlorothalonil or mancozeb.
        """)
    elif predicted_class == "Late_Blight":
        st.info("""
        **Late Blight**
        - **Symptoms**: Water-soaked lesions, white mold under leaves.
        - **Prevention**: Remove infected plants, improve air circulation.
        - **Cure**: Use copper-based fungicides or systemic fungicides.
        """)

# === AUTH SYSTEM ===
def login_signup_ui():
    st.title("🥔 Potato Disease Detection Login")
    choice = st.radio("Select", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if choice == "Login":
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        if st.button("Sign Up"):
            if register_user(username, password):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_signup_ui()
    st.stop()

# === HEADER ===
st.markdown('<div class="title">🥔 Potato Disease Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtext">Welcome, {st.session_state.username}</div>', unsafe_allow_html=True)

# === IMAGE PROCESSING ===
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def is_potato_leaf(pred):
    return np.max(pred) >= 0.7

def show_result(pred, img_source):
    predicted_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)

    if not is_potato_leaf(pred):
        st.error("⚠️ This doesn't appear to be a potato leaf.")
        return

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    fig = px.bar(
        x=class_names,
        y=pred * 100,
        labels={'x': 'Class', 'y': 'Probability %'},
        color=class_names,
        color_discrete_map={'Early_Blight': '#FFA500', 'Healthy': '#4CAF50', 'Late_Blight': '#FF6347'}
    )
    st.plotly_chart(fig, use_container_width=True)

    disease_info(predicted_class)

    prediction_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": predicted_class,
        "confidence": confidence,
        "source": img_source
    }
    save_prediction(st.session_state.username, convert_np(prediction_data))

# === TABS ===
tabs = st.tabs(["\U0001F4C1 Upload", "\U0001F4F7 Camera", "\U0001F4DC History", "\U0001F464 Account", "\U0001F6AA Logout"])

with tabs[0]:
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        if st.button("Predict"):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]
            show_result(prediction, "upload")

with tabs[1]:
    st.markdown("### Click below to open camera")
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if not st.session_state.camera_on:
        if st.button("📸 Open Camera"):
            st.session_state.camera_on = True
    else:
        camera_image = st.camera_input("Capture image")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", width=300)
            if st.button("Predict from Camera"):
                processed = preprocess_image(image)
                prediction = model.predict(processed)[0]
                show_result(prediction, "camera")
        if st.button("❌ Close Camera"):
            st.session_state.camera_on = False

with tabs[2]:
    history = load_json(HISTORY_DB)
    user_history = history.get(st.session_state.username, [])
    if user_history:
        st.subheader("Prediction History")
        for item in reversed(user_history):
            st.markdown(f"**Time:** {item['time']}  |  **Prediction:** {item['prediction']}  |  **Confidence:** {item['confidence']:.2f}%")
    else:
        st.info("No history available.")

with tabs[3]:
    st.subheader("Account Details")
    st.markdown(f"**Username:** {st.session_state.username}")

with tabs[4]:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        del st.session_state.username
        st.rerun()

# === FOOTER ===
st.markdown('<div class="footer">This app is made with ❤️ by <strong>Muhammad Junaid Khan</strong></div>', unsafe_allow_html=True)
