import os
from tensorflow.keras.models import load_model

def load_models():
    model_path = "models"

    return {
        "ResNet50": load_model(os.path.join(model_path, "tomato_stage_model_resnet50.keras")),
        "MobileNetV2": load_model(os.path.join(model_path, "tomato_stage_model.keras")),
        "DenseNet121": load_model(os.path.join(model_path, "tomato_stage_model_densenet121.keras")),
        "EfficientNetB0": load_model(os.path.join(model_path, "tomato_stage_model_efficientnetb0.keras"))
    }