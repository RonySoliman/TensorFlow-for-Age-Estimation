import cv2
import numpy as np
from face_detection import detect_faces_from_image
from feature_extraction import detect_mask
from age_estimation import estimate_age

# Outfit comparison with reduced resolution
def get_outfit_signature(image, bbox):
    """Generate outfit signature using downsampled region"""
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    
    # Get region below face (smaller area)
    y_start = min(y2 + 5, height-1)
    y_end = min(y2 + 50, height-1)
    
    if y_start >= y_end or (x2 - x1) < 10:
        return None
    
    # Extract and downsample region
    region = image[y_start:y_end, x1:x2]
    region = cv2.resize(region, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Use average color
    return np.mean(region, axis=(0, 1))

def compare_outfits(sig1, sig2, threshold=20):
    """Compare outfit signatures with tolerance"""
    if sig1 is None or sig2 is None:
        return False
    return np.abs(sig1 - sig2).mean() < threshold

def process_frame(frame, known_outfits, frame_index=None, outfit_threshold=20):
    # Detect faces
    _, face_locs = detect_faces_from_image(frame, max_dimension=640)  # Limit resolution
    
    annotated_frame = frame.copy()
    
    for i, loc in enumerate(face_locs):
        top, right, bottom, left = loc
        face_img = frame[top:bottom, left:right]
        
        if face_img.size == 0:
            continue
            
        # Get predictions
        mask_status = detect_mask(face_img)
        age = estimate_age(face_img)
        
        # Outfit detection
        outfit_signature = get_outfit_signature(frame, (left, top, right, bottom))
        outfit_is_new = True
        
        if outfit_signature is not None:
            for known_sig in known_outfits:
                if compare_outfits(outfit_signature, known_sig, outfit_threshold):
                    outfit_is_new = False
                    break
            if outfit_is_new:
                known_outfits.append(outfit_signature)
        
        # Draw annotations
        label = f"{mask_status} | {age}"
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.putText(annotated_frame, label, (left, top-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return annotated_frame, False  # Always return False for new outfit tracking
