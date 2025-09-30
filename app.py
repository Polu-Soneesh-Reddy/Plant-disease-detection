from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from datetime import datetime
import json
import plotly
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import load_model
from utils.segmentation import preprocess_image
import base64
from PIL import Image
from io import BytesIO
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure folders with absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'static', 'history.json')
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'best_model.keras')

# Ensure directories exist with proper permissions
for directory in [UPLOAD_FOLDER, os.path.dirname(HISTORY_FILE)]:
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o755)
        logger.info(f"Created directory: {directory}")
    else:
        # Ensure directory has proper permissions
        os.chmod(directory, 0o755)
        logger.info(f"Updated permissions for directory: {directory}")

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)
    logger.info(f"Created history file: {HISTORY_FILE}")
    # Ensure file has proper permissions
    os.chmod(HISTORY_FILE, 0o644)

try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    class_names = ['Healthy', 'Powdery', 'Rust']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def save_prediction_history(image_path, prediction, confidence):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': image_path,
            'prediction': prediction,
            'confidence': float(confidence)
        })
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
        logger.info(f"Saved prediction to history: {prediction}")
    except Exception as e:
        logger.error(f"Error saving prediction history: {str(e)}")

def generate_analytics():
    try:
        if not os.path.exists(HISTORY_FILE):
            return None, None
        
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        if not history:
            return None, None
            
        df = pd.DataFrame(history)
        
        # Create prediction distribution chart
        fig1 = px.pie(df, names='prediction', title='Distribution of Predictions')
        chart1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create confidence trend chart
        fig2 = px.line(df, x='timestamp', y='confidence', color='prediction',
                       title='Prediction Confidence Over Time')
        chart2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        
        return chart1, chart2
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return None, None
    
@app.route('/')
def home():
    pie_chart = {
        'data': [{'labels': ['Healthy', 'Rust', 'Powdery'], 'values': [30, 20, 50], 'type': 'pie'}],
        'layout': {'title': 'Disease Distribution'}
    }
    trend_chart = {
        'data': [{'x': ['Day 1', 'Day 2'], 'y': [10, 15], 'type': 'scatter'}],
        'layout': {'title': 'Prediction Trends'}
    }
    return render_template('index.html', pie_chart=pie_chart, trend_chart=trend_chart)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded. Please ensure the model file exists.'}), 500
        
    try:
        logger.info("Received prediction request")
        logger.debug(f"Request files: {request.files}")
        logger.debug(f"Request form: {request.form}")
        logger.debug(f"Request json: {request.json}")
        
        # Handle multipart form data
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                logger.error("No file selected")
                return jsonify({'error': 'No file selected'}), 400
                
            # Save the image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f'plant_{timestamp}.jpg'
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            
            try:
                file.save(image_path)
                logger.info(f"Saved image to: {image_path}")
            except Exception as e:
                logger.error(f"Error saving file: {str(e)}")
                return jsonify({'error': f'Error saving file: {str(e)}'}), 500
            
        # Handle JSON payload with base64 image
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            if ',' not in image_data:
                logger.error("Invalid image data format")
                return jsonify({'error': 'Invalid image data format'}), 400
                
            try:
                # Extract base64 data after the comma
                image_data = image_data.split(',')[1]
                img_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_bytes))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Save the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_filename = f'plant_{timestamp}.jpg'
                image_path = os.path.join(UPLOAD_FOLDER, image_filename)
                img.save(image_path, 'JPEG')
                logger.info(f"Saved image to: {image_path}")
                
            except Exception as e:
                logger.error(f"Error decoding image: {str(e)}")
                return jsonify({'error': 'Invalid image data'}), 400
        else:
            logger.error("No image data received")
            return jsonify({'error': 'No image data received'}), 400
        
        # Preprocess the image using the utility function
        try:
            img_array = preprocess_image(image_path)
            logger.info("Image preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return jsonify({'error': 'Error preprocessing image'}), 500
        
        # Make prediction
        try:
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            logger.info(f"Prediction: {predicted_class}, Confidence: {confidence}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({'error': 'Error making prediction'}), 500
        
        # Save prediction history
        try:
            save_prediction_history(image_path, predicted_class, confidence)
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
            # Don't return error here as the prediction was successful
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'image_url': f'/static/uploads/{image_filename}'
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/history')
def history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify([])

@app.route('/update_analytics')
def update_analytics():
    try:
        pie_chart, trend_chart = generate_analytics()
        return jsonify({
            'pie_chart': pie_chart,
            'trend_chart': trend_chart
        })
    except Exception as e:
        logger.error(f"Error updating analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting server with upload folder: {UPLOAD_FOLDER}")
    app.run(debug=True)
