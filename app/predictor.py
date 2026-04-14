import numpy as np
import cv2
from app.model_loader import load_models

# Different preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_pre
from tensorflow.keras.applications.densenet import preprocess_input as dense_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

models = load_models()

# ⚠️ VERY IMPORTANT — MUST MATCH TRAINING
class_names = ["breaker", "green", "pink", "red", "turning"]


def preprocess(img, model_name):
    img = cv2.resize(img, (224, 224))

    if model_name == "ResNet50":
        return resnet_pre(img)
    elif model_name == "MobileNetV2":
        return mobile_pre(img)
    elif model_name == "DenseNet121":
        return dense_pre(img)
    elif model_name == "EfficientNetB0":
        return eff_pre(img)

    return img / 255.0


def get_recommendation(stage):
    if stage == "green":
        return "Tomato is immature. Harvest not recommended.", "7–10 days"
    elif stage == "breaker":
        return "Early ripening stage. Can harvest soon.", "5–7 days"
    elif stage == "turning":
        return "Partially ripe. Suitable for transport.", "3–5 days"
    elif stage == "pink":
        return "Almost ripe. Ready for harvest.", "1–3 days"
    elif stage == "red":
        return "Fully ripe. Harvest immediately.", "0 days"
    else:
        return "Unknown", "N/A"


def predict_all(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not loaded")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = {}

    for name, model in models.items():

        processed = preprocess(img.copy(), name)
        processed = np.expand_dims(processed, axis=0)

        preds = model.predict(processed, verbose=0)[0]

        index = int(np.argmax(preds))
        label = class_names[index]
        confidence = float(np.max(preds)) * 100

        rec, days = get_recommendation(label)

        results[name] = {
            "label": label,
            "confidence": round(confidence, 2),
            "recommendation": rec,
            "days": days
        }

    return results