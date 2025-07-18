import cv2
import numpy as np
import os

# Global model with lazy loading
age_net = None

def load_age_model():
    global age_net
    if age_net is not None:
        return
    
    try:
        prototxt = "./models/deploy_age.prototxt"
        caffemodel = "./models/age_net.caffemodel"
        
        # Verify model exists
        if not all(os.path.exists(p) for p in [prototxt, caffemodel]):
            raise FileNotFoundError("Model files missing")
            
        age_net = cv2.dnn.readNet(prototxt, caffemodel)
        print("Age model loaded")
    except Exception as e:
        print(f"Age model error: {str(e)}")
        age_net = None

def estimate_age(face_img):
    if age_net is None:
        load_age_model()
        if age_net is None:
            return "N/A"
    
    try:
        # Use smaller input size
        blob = cv2.dnn.blobFromImage(
            face_img, 
            scalefactor=1.0,
            size=(150, 150),  # Reduced from 227
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        
        age_net.setInput(blob)
        preds = age_net.forward()
        return ["(0-15)", "(16-22)", "(23-29)", "(30-35)", 
                "(36-42)", "(43-50)", "(51-64)", "(65-80)"][preds[0].argmax()]
        
    except Exception:
        return "N/A"
