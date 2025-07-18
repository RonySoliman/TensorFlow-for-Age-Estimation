import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import numpy as np
from matplotlib import pyplot as plt
import shutil
from face_detection import detect_faces
from feature_extraction import detect_mask, extract_features
from face_matching import match_faces
from age_estimation import estimate_age  # Age estimation import

# Load database of known faces
database = {}
embedding_dir = "embeddings"
if os.path.exists(embedding_dir):
    for file in os.listdir(embedding_dir):
        if file.endswith('.npy'):
            person_id = os.path.splitext(file)[0]
            database[person_id] = np.load(os.path.join(embedding_dir, file))
    print(f"Loaded {len(database)} known face embeddings")
else:
    print("Warning: No embeddings directory found. Face matching disabled.")

# Clear output directories to prevent duplicates
output_dirs = ["output", "output/masks"]
for d in output_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Process images
results = []
data_dir = "./data/video1/"
valid_images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in valid_images:
    img_path = os.path.join(data_dir, img_name)
    print(f"Processing: {img_name}")
    
    try:
        # Detect faces
        image, face_locs = detect_faces(img_path)
    except Exception as e:
        print(f"  Error detecting faces: {str(e)}")
        continue
    
    # Skip if no faces found
    if not face_locs:
        print(f"  No faces found in {img_name}")
        continue
        
    img_results = []
    
    # Process each detected face
    for i, loc in enumerate(face_locs):
        try:
            top, right, bottom, left = loc
            print(f"  Face {i+1} at position: top={top}, right={right}, bottom={bottom}, left={left}")

            # Validate face coordinates
            if bottom <= top or right <= left:
                print(f"    Invalid face coordinates, skipping")
                continue
                
            # Check face size
            if (bottom - top) < 10 or (right - left) < 10:
                print(f"    Face too small, skipping")
                continue
                
            # Extract face region
            face_img = image[top:bottom, left:right]
            
            # Skip empty face regions
            if face_img.size == 0:
                print(f"    Empty face region, skipping")
                continue
                
            # Get predictions
            mask_status = detect_mask(face_img)
            
            # Face recognition
            face_embedding = extract_features(face_img)
            identity = ""
            if face_embedding is not None and database:
                identity, confidence = match_faces(face_embedding, database)
                identity = f"{identity} ({confidence:.2f})"
            
            # Get age prediction
            age = estimate_age(face_img)
    
            # Update results dictionary
            img_results.append({
                "identity": identity,
                "mask": mask_status,
                "age": age,
                "location": (left, top, right-left, bottom-top)
            })
    
            print(f"    {identity} | {mask_status} | Age: {age}")
            
        except Exception as e:
            print(f"    Error processing face {i+1}: {str(e)}")
            continue
    
    results.append((img_name, img_results))

# Visualization: Create annotated output images
for img_name, faces in results:
    img_path = os.path.join(data_dir, img_name)
    if not os.path.exists(img_path):
        print(f"  Image not found: {img_path}")
        continue
        
    image = cv2.imread(img_path)
    if image is None:
        print(f"  Failed to read image: {img_path}")
        continue
        
    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    
    # Draw bounding boxes and annotations
    for i, face in enumerate(faces):
        x, y, w, h = face['location']
        
        # Draw rectangle
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add text annotation - CORRECTED TO INCLUDE AGE
        text = f"{face['identity']} | {face['mask']} | Age: {face['age']}"
        plt.text(x, y-10, text, color='white', backgroundcolor='black',
                fontsize=9, verticalalignment='top')
    
    plt.axis('off')
    output_path = f"output/{img_name}"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved annotated image: {output_path}")

# Save face images for masks
for img_name, faces in results:
    img_path = os.path.join(data_dir, img_name)
    if not os.path.exists(img_path):
        continue
        
    image = cv2.imread(img_path)
    if image is None:
        continue
        
    for i, face in enumerate(faces):
        x, y, w, h = face['location']
        face_img = image[y:y+h, x:x+w]
        
        # Skip empty regions
        if face_img.size == 0:
            continue
        
        # Save mask images if predicted as "Mask"
        if face['mask'] == "Mask":
            mask_path = f"output/masks/{img_name}_face{i}.jpg"
            cv2.imwrite(mask_path, face_img)

# Print performance summary
total_faces = sum(len(faces) for _, faces in results)
print(f"\n{'='*50}")
print("PROCESSING SUMMARY")
print('='*50)
print(f"Total images processed: {len(results)}")
print(f"Total faces detected: {total_faces}")
print('='*50)
print("Done!")