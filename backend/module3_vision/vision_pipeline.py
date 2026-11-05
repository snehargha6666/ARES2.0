import os
import json
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module3_vision.cnn_classifier import predict_severity
from module3_vision.yolo_detector import detect_objects
from module3_vision.unet_segmenter import generate_mask

def run_vision_pipeline(image_path, cnn_weights=None, yolo_weights='yolov8n.pt', unet_weights=None):
    """Runs the complete vision analysis pipeline on a single image."""
    if not os.path.exists(image_path):
        return {'error': f'Input image not found at {image_path}'}

    # 1. Run CNN Severity Classifier
    cnn_kwargs = {}
    if cnn_weights:
        cnn_kwargs['weights_path'] = cnn_weights
    severity_results = predict_severity(image_path, **cnn_kwargs)

    # 2. Run YOLO Object Detector
    if not os.path.isabs(yolo_weights):
        yolo_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), yolo_weights)
    detector_results = detect_objects(image_path, weights_path=yolo_weights)

    # 3. Run U-Net Semantic Segmenter
    unet_kwargs = {}
    if unet_weights:
        unet_kwargs['weights_path'] = unet_weights
    segmentation_results = generate_mask(image_path, **unet_kwargs)

    # Combine pipeline output
    pipeline_results = {
        'image_path': image_path,
        'severity_analysis': severity_results,
        'object_detection': detector_results,
        'segmentation': segmentation_results
    }
    return pipeline_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARES 2.0 Vision Pipeline")
    parser.add_argument('--image', type=str, required=True, help="Path to the disaster image")
    args = parser.parse_args()

    final_output = run_vision_pipeline(args.image)
    print(json.dumps(final_output))
