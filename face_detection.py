import cv2
import mediapipe as mp

def detect_faces_from_image(
    image, 
    max_dimension=640,  # Reduced from 2000
    model_selection=0,   # Faster model (0=short-range)
    min_detection_confidence=0.3  # Lower confidence threshold
):
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    )
    
    # Always downscale large images
    height, width = image.shape[:2]
    scale = 1.0
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image.copy()
        new_height, new_width = height, width
    
    # Convert and process
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    face_locations = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            xmin = int(bbox.xmin * new_width)
            ymin = int(bbox.ymin * new_height)
            box_width = int(bbox.width * new_width)
            box_height = int(bbox.height * new_height)
            
            top = max(0, ymin)
            left = max(0, xmin)
            bottom = min(new_height-1, ymin + box_height)
            right = min(new_width-1, xmin + box_width)
            
            # Scale coordinates
            if scale < 1.0:
                top = int(top / scale)
                left = int(left / scale)
                bottom = int(bottom / scale)
                right = int(right / scale)
            
            face_locations.append((top, right, bottom, left))
    
    return image, face_locations
