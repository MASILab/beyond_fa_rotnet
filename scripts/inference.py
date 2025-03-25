import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import json
import argparse
import os

# Author: Gaurav Rudravaram

class RotationRegressionModel(nn.Module):
    def __init__(self):
        super(RotationRegressionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc_size = self._calculate_fc_size()

        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def _calculate_fc_size(self):
        dummy_input = torch.zeros(1, 1, 64, 64, 64)
        output = self.conv_layers(dummy_input)
        return output.numel()

    def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, self.fc_size)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            return x


def preprocess_fa_map(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    if np.all(img == 0) or np.max(img) - np.min(img) < 1e-6:
        return None

    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(64, 64, 64), mode='trilinear', align_corners=False)
    img = img.squeeze().numpy()

    p1, p99 = np.percentile(img[img > 0], [1, 99])
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1)


def infer_rotation(model, fa_map_path, device, output_json_path):
    try:
        # Get output directory and filename
        output_dir = os.path.dirname(output_json_path)
        filename = os.path.basename(output_json_path)
        print(f"Output directory: {output_dir}")
        print(f"Output filename: {filename}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess the FA map
        img = nib.load(fa_map_path).get_fdata()
        img = preprocess_fa_map(img)
        if img is None:
            raise ValueError("Invalid FA map after preprocessing.")

        # Run inference
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output_tensor = model(img_tensor).squeeze(0).cpu().numpy()
        
        # Ensure the output is a 128x1 vector
        if output_tensor.shape != (128,):
            raise ValueError(f"Expected output shape (128,), got {output_tensor.shape}")
        
        # Convert to list and save as JSON with proper formatting
        output_list = output_tensor.tolist()
        with open(output_json_path, 'w') as json_file:
            json.dump(output_list, json_file, indent=4)
        
        print(f"Tensor output saved to {output_json_path}")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on FA map')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--fa_map', type=str, required=True, help='Path to FA map file')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save output JSON')
    
    args = parser.parse_args()

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RotationRegressionModel().to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))

    # Perform inference and save the output tensor as JSON
    infer_rotation(model, args.fa_map, device, args.output_json)
