import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Define the CNN architecture for Disaster Severity Classification (Low, Medium, Severe)
class DisasterCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DisasterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define classes
CLASSES = ['Low', 'Medium', 'Severe']

# Instantiate model placeholder
model = DisasterCNN(num_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def load_model_weights(weights_path):
    """Loads model weights if they exist, otherwise uses randomly initialized weights."""
    global model
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded CNN model weights from {weights_path}", file=sys.stderr)
        return True
    else:
        print(f"Weights file not found at {weights_path}.", file=sys.stderr)
        return False

def predict_severity(image_path, weights_path=None):
    """Processes an input image, performs inference, and returns predicted severity with confidence."""
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.pth')
    # Load weights for prediction
    if not load_model_weights(weights_path):
        return {
            'severity': 'Unavailable',
            'confidence': 0.0,
            'warning': 'CNN weights are not installed; no image severity prediction was produced.'
        }
    
    # Standard preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        confidence, class_idx = torch.max(probabilities, dim=0)
        
        return {
            'severity': CLASSES[class_idx.item()],
            'confidence': float(confidence.item())
        }
    except Exception as e:
        return {
            'error': str(e)
        }
