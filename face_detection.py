import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_faces_from_image(image, max_dimension=2000):
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Convert to RGB for MediaPipe
        rgb_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_resized)
        
        original_face_locations = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Convert relative coordinates to absolute (resized image)
                xmin = int(bbox.xmin * new_width)
                ymin = int(bbox.ymin * new_height)
                bbox_width = int(bbox.width * new_width)
                bbox_height = int(bbox.height * new_height)
                
                # Convert to (top, right, bottom, left) format
                top = ymin
                bottom = ymin + bbox_height
                left = xmin
                right = xmin + bbox_width
                
                # Scale coordinates back to original image
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                
                # Clamp to image boundaries
                top = max(0, top)
                bottom = min(height, bottom)
                left = max(0, left)
                right = min(width, right)
                
                original_face_locations.append((top, right, bottom, left))
                
        return image, original_face_locations
        
    else:
        # Process full-size image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_image)
        
        face_locations = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Convert relative coordinates to absolute
                xmin = int(bbox.xmin * width)
                ymin = int(bbox.ymin * height)
                bbox_width = int(bbox.width * width)
                bbox_height = int(bbox.height * height)
                
                top = ymin
                bottom = ymin + bbox_height
                left = xmin
                right = xmin + bbox_width
                
                # Clamp to image boundaries
                top = max(0, top)
                bottom = min(height, bottom)
                left = max(0, left)
                right = min(width, right)
                
                face_locations.append((top, right, bottom, left))
                
        return image, face_locations
