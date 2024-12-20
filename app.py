from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import cv2

# Flask App Initialize
app = Flask(__name__)

# Model Load
model = load_model("models/emotion_model.keras")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
                frame = cv2.imread(file_path)  # Read the image using OpenCV
                if frame is None:
                    error = "Error: Could not open or find the image."
                    return render_template("index.html", result=None, error=error)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
                faces = face_cascade.detectMultiScale(gray, 1.3, 3)  # Detect faces

                for (x, y, w, h) in faces:
                    sub_face_img = gray[y:y+h, x:x+w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]

                    # Draw rectangle around the face and put the label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)  # Background for text
                    cv2.putText(frame, emotion_labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Save the output image with the rectangle and label
                output_image_path = os.path.join("static/uploads", "output_" + file.filename)
                cv2.imwrite(output_image_path, frame)

                return render_template("index.html", result=emotion_labels[label], image="output_" + file.filename)

            except UnidentifiedImageError:
                error = "Invalid image format. Please upload a valid image file."
            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template("index.html", result=None, error=error)

# Route for Live Emotion Detection
@app.route("/live")
def live_emotion_detection():
    return render_template("live.html")

# Route to generate frames for live emotion detection
def generate_frames():
    cap = cv2.VideoCapture(0)  # Start webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI (Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to model input size
            roi_gray = roi_gray / 255.0  # Normalize the image
            roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
            roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
            
            # Predict emotion
            predictions = model.predict(roi_gray)
            emotion_index = np.argmax(predictions)  # Get the index of max value
            emotion_label = emotion_labels[emotion_index]
            
            # Draw a rectangle around the face and display emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Rectangle around face
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)  # Display label
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)