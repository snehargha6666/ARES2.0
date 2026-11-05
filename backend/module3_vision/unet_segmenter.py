import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512)
        ])
        
        # Bottleneck channels doubled to 1024
        self.bottleneck = DoubleConv(512, 1024)
        
        # Upsampling channel dimensions correctly aligned
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Handle dimension padding if sizes don't perfectly match
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_x)

        return self.final_conv(x)

model = UNet(in_channels=3, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def load_unet_weights(weights_path):
    global model
    if not os.path.exists(weights_path):
        return False
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return True

def estimate_flood_mask(image):
    """Estimate connected floodwater regions when trained U-Net weights are unavailable."""
    rgb = np.asarray(image).astype(np.float32) / 255.0
    brightness = np.mean(rgb, axis=2)
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    height = rgb.shape[0]
    vertical_prior = np.indices(rgb.shape[:2])[0] > height * 0.28
    water_like = (
        (saturation < 0.22) &
        (brightness > 0.22) &
        (brightness < 0.82) &
        vertical_prior
    )
    mask = (water_like.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(80, int(mask.shape[0] * mask.shape[1] * 0.002))
    clean = np.zeros_like(mask)
    for component in range(1, components):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == component] = 255
    return clean

def generate_flood_fallback(image):
    mask = estimate_flood_mask(image)
    overlay = np.asarray(image).copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.45 + np.array([45, 125, 235]) * 0.55
    ).astype(np.uint8)
    return Image.fromarray(overlay)

def generate_mask(image_path, weights_path=None, output_dir=None):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if weights_path is None:
        weights_path = os.path.join(module_dir, 'unet_weights.pth')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(module_dir), 'public', 'masks')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        original_image = Image.open(image_path).convert('RGB')
        orig_size = original_image.size
        has_weights = load_unet_weights(weights_path)
        if has_weights:
            input_tensor = preprocess(original_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            mask_img = Image.fromarray((probs > 0.5).astype(np.uint8) * 255)
            method = "U-Net"
        else:
            mask_img = generate_flood_fallback(original_image.resize((256, 256)))
            method = "classical flood-region estimate (U-Net weights unavailable)"
        mask_img = mask_img.resize(orig_size, Image.NEAREST)
        os.makedirs(output_dir, exist_ok=True)
        mask_filename = f"mask_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        save_path = os.path.join(output_dir, mask_filename)
        mask_img.save(save_path)
        return {
            'mask_path': os.path.join('public/masks', mask_filename).replace("\\", "/"),
            'method': method
        }
    except Exception as e:
        return {'error': str(e)}
