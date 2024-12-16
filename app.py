from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

# Flask App Initialize
app = Flask(__name__)

# Model Load
model = load_model("models/emotion_model.keras")

# Route for Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            try:
                # File Save
                file_path = os.path.join("static/uploads", file.filename)
                file.save(file_path)

                # Image Processing
                img = Image.open(file_path).convert("L")  # Convert to Grayscale
                img = img.resize((48, 48))               # Resize to 48x48
                img = np.array(img) / 255.0              # Normalize
                img = img.reshape(1, 48, 48, 1)

                # Prediction
                predictions = model.predict(img)
                emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
                predicted_label = emotion_labels[np.argmax(predictions)]

                return render_template("index.html", result=predicted_label, image=file.filename)

            except UnidentifiedImageError:
                error = "Invalid image format. Please upload a valid image file."
            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template("index.html", result=None, error=error)

if __name__ == "__main__":
    app.run(debug=True)
