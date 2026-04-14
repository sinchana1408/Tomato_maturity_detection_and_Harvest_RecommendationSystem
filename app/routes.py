from flask import Blueprint, render_template, request
import os
from app.predictor import predict_all

main = Blueprint('main', __name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@main.route('/')
def home():
    return render_template("index.html")


@main.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        results = predict_all(filepath)
        best_model = max(results, key=lambda x: results[x]['confidence'])

        return render_template(
            "result.html",
            image="/" + filepath,
            results=results,
            best_model=best_model
        )


@main.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")