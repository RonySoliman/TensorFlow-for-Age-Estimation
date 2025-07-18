# feature_extraction.py
import cv2
import numpy as np
import os
import logging
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and tracker
mask_model = None
performance_tracker = None
target_height, target_width = 224, 224
model_type = "sigmoid"

# Lightweight model loader
def load_mask_model():
    global mask_model, target_height, target_width, model_type
    
    if mask_model is not None:
        return True
        
    try:
        from tensorflow.keras.models import load_model
        logger.info("Loading mask detection model...")
        start_time = time.time()
        
        # Load model with reduced verbosity
        mask_model = load_model('./models/mask_detector.h5', compile=False)
        
        # Get model input dimensions
        if len(mask_model.input_shape) == 4:
            _, target_height, target_width, _ = mask_model.input_shape
        else:
            # Handle different input shapes
            target_height, target_width = 224, 224
            
        # Determine model output type
        output_shape = mask_model.output_shape
        if len(output_shape) == 2 and output_shape[1] == 2:
            model_type = "softmax"
        else:
            model_type = "sigmoid"
            
        logger.info(f"Mask model loaded in {time.time()-start_time:.2f}s")
        logger.info(f"Input size: {target_height}x{target_width}, Output type: {model_type}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading mask model: {str(e)}")
        mask_model = None
        return False

# Model performance tracking
class ModelPerformanceTracker:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []
    
    def add_prediction(self, true_label, pred_label):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)
    
    def generate_report(self):
        if not self.true_labels:
            return "No predictions made yet"
        
        try:
            report = classification_report(
                self.true_labels,
                self.pred_labels,
                target_names=["No Mask", "Mask"],
                output_dict=True
            )
            
            # Generate confusion matrix
            cm = confusion_matrix(self.true_labels, self.pred_labels)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["No Mask", "Mask"],
                        yticklabels=["No Mask", "Mask"])
            plt.title("Mask Detection Confusion Matrix")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig("mask_confusion_matrix.png", dpi=120, bbox_inches='tight')
            plt.close()
            
            return report
        except Exception as e:
            return f"Report generation error: {str(e)}"

# Initialize performance tracker
performance_tracker = ModelPerformanceTracker()

def detect_mask(face_image, true_label=None):
    global performance_tracker
    
    # Lazy load model
    if not load_mask_model():
        return "Model Error"
    
    try:
        # Efficient preprocessing
        resized_face = cv2.resize(face_image, (target_width, target_height))
        normalized = resized_face.astype("float32") / 255.0
        processed = np.expand_dims(normalized, axis=0)
        
        # Prediction
        prediction = mask_model.predict(processed, verbose=0)
        
        # Interpret results
        if model_type == "softmax":
            mask_prob = prediction[0][1]
            pred_label = "Mask" if mask_prob > 0.5 else "No Mask"
        else:
            mask_prob = prediction[0][0]
            pred_label = "Mask" if mask_prob > 0.5 else "No Mask"
        
        # Track performance
        if true_label is not None:
            performance_tracker.add_prediction(true_label, pred_label)
            
        return pred_label
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error"

def get_performance_metrics():
    """Return current performance metrics"""
    try:
        return performance_tracker.generate_report()
    except Exception as e:
        return {"error": f"Metrics error: {str(e)}"}
