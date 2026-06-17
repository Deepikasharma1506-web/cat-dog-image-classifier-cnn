from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("cat_dog_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(150,150)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "DOG 🐶"
    else:
        result = "CAT 🐱"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence*100,2)
    )

if __name__ == "__main__":
    app.run(debug=True)