import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from utils.segmentation import LeafSegmenter
from flask import Flask, request, jsonify  # <-- added Flask imports
import logging

class PlantDiseasePredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.segmenter = LeafSegmenter()
            self.class_names = ['healthy', 'powdery', 'rust']
            self.img_size = (256, 256)
            self.leaf_threshold = 0.1
            self.confidence_thresholds = {
                'healthy': 0.65,
                'powdery': 0.45,
                'rust': 0.50
            }
        except Exception as e:
            raise RuntimeError(f"Predictor initialization failed: {str(e)}")

    def _is_leaf_image(self, image_norm):
        hsv = cv2.cvtColor((image_norm * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.count_nonzero(green_mask)
        total_pixels = green_mask.size
        green_percentage = green_pixels / total_pixels
        return green_percentage > self.leaf_threshold

    def _preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            return img / 255.0
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {str(e)}")

    def predict(self, image_path, leaf_index=None, explain=False, all_leaves=False):
        try:
            image_norm = self._preprocess_image(image_path)

            if not self._is_leaf_image(image_norm):
                return {
                    'error': 'no_leaves',
                    'message': 'No significant leaf area detected in the image'
                }

            masks = self.segmenter.segment((image_norm * 255).astype(np.uint8))

            if masks is None or masks.size == 0:
                return {
                    'error': 'no_leaves',
                    'message': 'No individual leaves could be identified'
                }

            self.segmenter.masks = masks

            if not all_leaves and leaf_index is None:
                leaf_index = 0

            if leaf_index is not None:
                if leaf_index < 0 or leaf_index >= len(masks):
                    return {
                        'error': 'invalid_leaf',
                        'message': f'Leaf index {leaf_index} is out of range (0-{len(masks)-1})'
                    }
                masks = [masks[leaf_index]]

            results = []
            for i, mask in enumerate(masks):
                leaf_img = self.segmenter.extract_leaf(image_norm, mask)
                leaf_img = cv2.resize(leaf_img, self.img_size)
                pred = self.model.predict(np.expand_dims(leaf_img, 0), verbose=0)[0]

                pred_class_idx = np.argmax(pred)
                pred_class = self.class_names[pred_class_idx]
                confidence = float(pred[pred_class_idx])

                if pred_class == 'healthy' and confidence < self.confidence_thresholds['healthy']:
                    powdery_conf = float(pred[1])
                    rust_conf = float(pred[2])

                    if powdery_conf > self.confidence_thresholds['powdery']:
                        pred_class = 'powdery'
                        confidence = powdery_conf
                    elif rust_conf > self.confidence_thresholds['rust']:
                        pred_class = 'rust'
                        confidence = rust_conf

                result = {
                    'leaf_index': i,
                    'class': pred_class,
                    'confidence': confidence,
                    'probabilities': {
                        cls: float(prob) for cls, prob in zip(self.class_names, pred)
                    }
                }

                results.append(result)

            return self._aggregate_results(results)
        except Exception as e:
            return {'error': 'processing_error', 'message': str(e)}

    def _aggregate_results(self, results):
        try:
            if not results:
                return {'error': 'no_results', 'message': 'No prediction results available'}

            return {
                'primary_class': results[0]['class'],
                'confidence': results[0]['confidence'],
                'per_leaf': results,
                'leaf_count': len(results)
            }
        except Exception as e:
            return {'error': 'aggregation_error', 'message': str(e)}

# Flask route for prediction
app = Flask(__name__)
predictor = PlantDiseasePredictor('models/best_model.keras')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400

        image_file = request.files['image']
        temp_path = 'temp_upload.jpg'
        image_file.save(temp_path)

        result = predictor.predict(temp_path, all_leaves=True)
        os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        logging.exception("Unexpected error in predict route")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
