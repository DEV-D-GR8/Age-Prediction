# Real-Time Age Prediction with OpenCV and Flask

Welcome to the Real-Time Age Prediction project repository! This project combines the power of Convolutional Neural Networks (CNN) for multi-class age classification with real-time face detection using OpenCV. The Flask web framework is employed to deploy the model and create an interactive user interface for live age prediction.

## Tech Stack

- **Deep Learning Framework:** TensorFlow
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Computer Vision Library:** OpenCV
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Backend:** Python

## Project Overview

### Dataset Refinement

The model is trained on a diverse and well-curated dataset of facial images, categorized into multiple age groups. Extensive data preprocessing and augmentation techniques are applied to enhance the model's performance and robustness.

### Real-Time Face Detection

The project utilizes OpenCV for real-time face detection. A bounding box is drawn around the detected face, providing a visual representation of the area used for age prediction.

### CNN Model

The CNN model is designed for multi-class age classification. The model is trained from scratch to predict age groups in real-time.

### Flask Web Application

The Flask web application seamlessly integrates the real-time face detection and age prediction functionalities. Users can view live video streams from their device's camera, with bounding boxes drawn around detected faces and corresponding age predictions displayed above.

## Setup and Testing

To run the Real-Time Age Prediction project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DEV-D-GR8/Age-Prediction.git
   cd Age-Prediction

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Application:**
   ```bash
   python App/app.py

4. **Visit the Application:**
Open your web browser and navigate to [http://localhost:5000](http://localhost:5000).

5. **Test the Real-Time Prediction:**
Access the webcam through the web interface and observe real-time face detection with age predictions displayed above the detected faces.

## License

This project is licensed under the [MIT License](LICENSE).

   





















   
