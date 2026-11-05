from ultralytics import YOLO
import os
import sys
from PIL import Image
import cv2
import numpy as np
from module3_vision.unet_segmenter import estimate_flood_mask

# Initialize YOLOv8 model placeholder (e.g., using yolov8n.pt or a custom-trained model)
# It will auto-download 'yolov8n.pt' if a custom weights path doesn't exist
model = None

def load_yolo_model(weights_path='yolov8n.pt'):
    """Loads the YOLOv8 model from the given path."""
    global model
    if model is None:
        model = YOLO(weights_path)
        print(f"YOLO model loaded using weights: {weights_path}", file=sys.stderr)
    return model

def detect_objects(image_path, weights_path='yolov8n.pt', confidence_threshold=0.45):
    """Runs YOLOv8 object detection on an image and returns structured detections."""
    try:
        yolo_model = load_yolo_model(weights_path)
        
        # Run inference
        results = yolo_model(image_path, conf=confidence_threshold, verbose=False)
        source_image = Image.open(image_path).convert('RGB')
        water_mask = estimate_flood_mask(source_image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates (x_min, y_min, x_max, y_max)
                xyxy = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = xyxy
                
                # Get class label
                class_idx = int(box.cls[0].item())
                label = yolo_model.names[class_idx]
                
                # Get confidence
                confidence = float(box.conf[0].item())
                box_mask = water_mask[
                    max(0, int(y_min)):min(water_mask.shape[0], int(y_max)),
                    max(0, int(x_min)):min(water_mask.shape[1], int(x_max))
                ]
                water_overlap = float(np.mean(box_mask > 0)) if box_mask.size else 0.0
                if label == "boat" and water_overlap < 0.2:
                    continue
                
                detections.append({
                    'label': label,
                    'box': [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                    'confidence': round(confidence, 4)
                    ,'water_overlap': round(water_overlap, 4)
                })

        annotated_filename = f"annotated_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        annotated_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "masks", annotated_filename)
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        annotated = cv2.cvtColor(np.asarray(source_image), cv2.COLOR_RGB2BGR)
        for detection in detections:
            x_min, y_min, x_max, y_max = [int(value) for value in detection['box']]
            label = f"{detection['label']} {detection['confidence']:.2f}"
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 220, 80), 2)
            cv2.putText(
                annotated,
                label,
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 80),
                2,
                cv2.LINE_AA
            )
        cv2.imwrite(annotated_path, annotated)

        return {
            'detections': detections,
            'annotated_image_path': os.path.join('public/masks', annotated_filename).replace("\\", "/"),
            'confidence_threshold': confidence_threshold,
            'model': os.path.basename(weights_path),
            'warning': 'This is a general COCO detector, not a disaster-specific model.'
        }
    except Exception as e:
        return {
            'error': str(e)
        }
