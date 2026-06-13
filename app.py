from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/car_model_classifier.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

classes = sorted(os.listdir("dataset"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = round(
        float(np.max(prediction)) * 100,
        2
    )

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=confidence,
        image="/" + filepath
    )

if __name__ == "__main__":
    app.run(debug=True)