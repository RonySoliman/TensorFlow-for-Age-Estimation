# face_detection.py
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces_from_image(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                box = [int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                       int(bboxC.width * iw), int(bboxC.height * ih)]
                boxes.append(box)
        return boxes
