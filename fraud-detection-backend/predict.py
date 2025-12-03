"""
Predict / Test Fraud Detection

Usage:
    python predict.py --source test.jpg
    python predict.py --source test_folder/
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
from ultralytics import YOLO

VALID_CLASSES = ['firstName', 'lastName', 'photo']
IMAGE_SIZE = 224
LATENT_DIM = 128


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
    def __init__(self, yolo_path='detect_objects.pt', models_folder='models', device='cpu'):
        self.device = device
        print(f"Loading YOLO: {yolo_path}")
        self.yolo = YOLO(yolo_path)
        
        self.models = {}
        self.thresholds = {}
        
        for cls in VALID_CLASSES:
            model_path = Path(models_folder) / cls / 'model.pth'
            config_path = Path(models_folder) / cls / 'config.json'
            
            if model_path.exists() and config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                model = FraudAutoencoder(config.get('latent_dim', LATENT_DIM)).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()
                
                self.models[cls] = model
                self.thresholds[cls] = config['threshold']
                print(f"  ✓ {cls}: threshold={config['threshold']:.6f}")
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def detect(self, image_path, output_folder='results'):
        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': f'Cannot read {image_path}'}
        
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
                label = f"{'FRAUD' if is_fraud else 'OK'}: {cls_name} ({ratio:.2f}x)"
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        verdict = "FRAUD DETECTED!" if fraud_detected else "AUTHENTIC"
        cv2.rectangle(img, (5,5), (350,55), (0,0,0), -1)
        cv2.putText(img, verdict, (15,42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                    (0,0,255) if fraud_detected else (0,255,0), 3)
        
        Path(output_folder).mkdir(exist_ok=True)
        out_file = Path(output_folder) / f"result_{Path(image_path).name}"
        cv2.imwrite(str(out_file), img)
        
        return {
            'fraud_detected': fraud_detected,
            'fraud_reasons': fraud_reasons,
            'detections': detections,
            'output_image': str(out_file)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', '-y', default='detect_objects.pt')
    parser.add_argument('--models', '-m', default='models')
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--output', '-o', default='results')
    parser.add_argument('--device', '-d', default='cpu')
    args = parser.parse_args()
    
    detector = FraudDetector(args.yolo, args.models, args.device)
    source = Path(args.source)
    
    if source.is_file():
        result = detector.detect(str(source), args.output)
        print(f"\n{'='*50}")
        print(f"{'🚨 FRAUD!' if result['fraud_detected'] else '✅ AUTHENTIC'}")
        for det in result['detections']:
            print(f"  {det['field']}: {det['error_ratio']:.2f}x {'❌' if det['is_fraud'] else '✓'}")
        print(f"Saved: {result['output_image']}")
    else:
        images = list(source.glob('*.jpg')) + list(source.glob('*.png'))
        fraud_count = sum(1 for img in images if detector.detect(str(img), args.output)['fraud_detected'])
        print(f"\nTotal: {len(images)} | Fraud: {fraud_count} | OK: {len(images)-fraud_count}")


if __name__ == '__main__':
    main()
