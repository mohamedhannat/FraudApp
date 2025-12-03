"""
Flask API Server - Local Fraud Detection

Usage:
    python server.py --port 5000

Endpoints:
    GET  /        - Info
    GET  /health  - Health check
    POST /detect  - Detect fraud {"image": "base64..."}
"""

import argparse
import json
import base64
import tempfile
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

VALID_CLASSES = ['firstName', 'lastName', 'photo']
IMAGE_SIZE = 224
LATENT_DIM = 128

app = Flask(__name__)
CORS(app)


class FraudAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.fc_encode = nn.Linear(512*7*7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512*7*7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        x = self.fc_decode(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.decoder(x)
    
    def get_error(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x))**2, dim=[1,2,3])


class FraudDetector:
    def __init__(self, yolo_path, models_folder, device='cpu'):
        self.device = device
        self.yolo = YOLO(yolo_path)
        self.models = {}
        self.thresholds = {}
        
        for cls in VALID_CLASSES:
            mp = Path(models_folder) / cls / 'model.pth'
            cp = Path(models_folder) / cls / 'config.json'
            if mp.exists() and cp.exists():
                with open(cp) as f:
                    cfg = json.load(f)
                model = FraudAutoencoder(cfg.get('latent_dim', LATENT_DIM)).to(device)
                model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
                model.eval()
                self.models[cls] = model
                self.thresholds[cls] = cfg['threshold']
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()
        ])
    
    def detect_base64(self, img_b64):
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(base64.b64decode(img_b64))
            tmp_path = tmp.name
        
        try:
            img = cv2.imread(tmp_path)
            if img is None:
                return {'success': False, 'error': 'Cannot read image'}
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.yolo.predict(source=img, save=False, device=self.device, conf=0.4, verbose=False)
            
            detections = []
            fraud_detected = False
            fraud_reasons = []
            
            for result in results:
                for box in result.boxes:
                    cls_name = self.yolo.names[int(box.cls[0].item())]
                    if cls_name not in VALID_CLASSES:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w = img.shape[:2]
                    crop = img_rgb[max(0,y1-3):min(h,y2+3), max(0,x1-3):min(w,x2+3)]
                    if crop.size == 0:
                        continue
                    
                    error = threshold = 0.0
                    is_fraud = False
                    
                    if cls_name in self.models:
                        tensor = self.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
                        error = self.models[cls_name].get_error(tensor).item()
                        threshold = self.thresholds[cls_name]
                        is_fraud = error > threshold
                        if is_fraud:
                            fraud_detected = True
                            fraud_reasons.append(f"{cls_name}: {error/threshold:.2f}x threshold")
                    
                    ratio = error / threshold if threshold > 0 else 0
                    detections.append({
                        'field': cls_name, 'recon_error': round(error, 6),
                        'threshold': round(threshold, 6), 'error_ratio': round(ratio, 3),
                        'is_fraud': is_fraud, 'bbox': [x1, y1, x2, y2]
                    })
                    
                    color = (0,0,255) if is_fraud else (0,255,0)
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
                    cv2.putText(img, f"{'FRAUD' if is_fraud else 'OK'}: {cls_name}", 
                               (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.rectangle(img, (5,5), (350,55), (0,0,0), -1)
            cv2.putText(img, "FRAUD!" if fraud_detected else "AUTHENTIC", (15,42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255) if fraud_detected else (0,255,0), 3)
            
            _, buf = cv2.imencode('.jpg', img)
            
            return {
                'success': True, 'fraud_detected': fraud_detected,
                'fraud_reasons': fraud_reasons, 'detections': detections,
                'result_image': base64.b64encode(buf).decode()
            }
        finally:
            os.unlink(tmp_path)


detector = None

@app.route('/')
def index():
    return jsonify({'service': 'Fraud Detection API', 'endpoints': ['GET /health', 'POST /detect']})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models': list(detector.models.keys()) if detector else []})

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image'}), 400
    return jsonify(detector.detect_base64(data['image']))


def main():
    global detector
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', '-y', default='detect_objects.pt')
    parser.add_argument('--models', '-m', default='models')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--device', '-d', default='cpu')
    args = parser.parse_args()
    
    print(f"Loading models...")
    detector = FraudDetector(args.yolo, args.models, args.device)
    print(f"\nServer: http://localhost:{args.port}")
    print(f"API: http://localhost:{args.port}/detect\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
