from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input as piRes
from keras.models import load_model

app = Flask(__name__)

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load age prediction model
model = load_model("model/best1.h5")
class_labels = ["1-15", "16-25", "26-30", "31-35", "36-45", "46-60", "60-116"]

def preprocess_image(face_image):
    face_image = cv2.resize(face_image, (224, 224))
    face_image = piRes(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def predict_age(preprocessed_face):
    predictions = model.predict(preprocessed_face)
    top_predicted_label = np.argmax(predictions)
    predicted_range = class_labels[top_predicted_label]
    return predicted_range

def display_age(image, face_coordinates, age_range):
    x, y, w, h = face_coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    font_color = (255, 0, 255)
    cv2.putText(
        image,
        f"Age: {age_range}",
        (x, y - 10),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for x, y, w, h in faces:
            face_image = frame[y : y + h, x : x + w]

            preprocessed_face = preprocess_image(face_image)

            age_category = predict_age(preprocessed_face)

            display_age(frame, (x, y, w, h), age_category)

        ret, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

age_prediction_enabled = False

@app.route('/start_stop_age_prediction')
def start_stop_age_prediction():
    global age_prediction_enabled
    age_prediction_enabled = not age_prediction_enabled
    return "Age Prediction Enabled" if age_prediction_enabled else "Age Prediction Disabled"

@app.route('/video_feed')
def video_feed():
    if age_prediction_enabled:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Age Prediction is Disabled"

if __name__ == '__main__':
    app.run(debug=True)
