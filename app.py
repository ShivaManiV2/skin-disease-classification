import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 96
DEVICE = torch.device("cpu")

CLASS_INFO = {
    "akiec": "Actinic Keratoses (Pre-cancerous lesion)",
    "bcc": "Basal Cell Carcinoma (Skin cancer)",
    "bkl": "Benign Keratosis (Non-cancerous)",
    "df": "Dermatofibroma (Benign skin lesion)",
    "mel": "Melanoma (Dangerous skin cancer)",
    "nv": "Melanocytic Nevus (Common mole)",
    "vasc": "Vascular Lesion (Blood vessel related)"
}

CLASS_KEYS = list(CLASS_INFO.keys())

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(
    page_title="Skin Disease AI",
    layout="wide"
)

st.title("🩺 Skin Disease Classification & Segmentation")
st.write("Deep Learning on HAM10000 Dataset")

# ==============================
# IMAGE TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD MODELS (EDIT PATHS)
# ==============================
@st.cache_resource
def load_mobilenet():
    model = models.mobilenet_v2(
        weights=MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.last_channel, 7)
    model.load_state_dict(torch.load(
        "classification/mobilenet_epoch_15.pth",
        map_location=DEVICE
    ))
    model.eval()
    return model


@st.cache_resource
def load_resnet():
    model = models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(
        "classification/resnet18_best.pth",
        map_location=DEVICE
    ))
    model.eval()
    return model

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Classification Model",
    ["MobileNetV2", "ResNet18"]
)

uploaded_file = st.file_uploader(
    "Upload Dermoscopic Image",
    type=["jpg", "png", "jpeg"]
)

# ==============================
# MAIN
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # preprocess
    input_tensor = transform(image).unsqueeze(0)

    # load selected model
    if model_choice == "MobileNetV2":
        model = load_mobilenet()
    else:
        model = load_resnet()

    # prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    with col2:
        st.subheader("🔍 Prediction")

        pred_key = CLASS_KEYS[pred_class]
        friendly_name = CLASS_INFO[pred_key]

        st.success(f"**Disease:** {friendly_name}")
        st.info(f"**Confidence:** {confidence:.2%}")

        st.subheader("📊 Class Probabilities")

        prob_dict = {
            CLASS_INFO[CLASS_KEYS[i]]: float(probs[0][i])
            for i in range(len(CLASS_KEYS))
        }

        st.bar_chart(prob_dict)

st.markdown("---")
st.caption("Developed using PyTorch & Streamlit")