"""
Crop Training Data from ID Cards
Extracts: firstName, lastName, photo using YOLO

Usage:
    python crop_data.py --yolo detect_objects.pt --source data/ --output cropped/
"""

import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

VALID_CLASSES = ['firstName', 'lastName', 'photo']


def crop_data(model_path, source_folder, output_folder, device='cpu', conf=0.4):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    print(f"YOLO classes: {list(model.names.values())}")
    print(f"Using ONLY: {VALID_CLASSES}\n")
    
    source_path = Path(source_folder)
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(list(source_path.glob(ext)))
    
    print(f"Found {len(images)} images")
    
    if not images:
        print(f"\nERROR: No images in {source_folder}")
        print("Add ID card images to the data/ folder")
        return None
    
    stats = {cls: 0 for cls in VALID_CLASSES}
    
    for img_file in tqdm(images, desc="Cropping"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        results = model.predict(source=img, save=False, device=device, conf=conf, verbose=False)
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                
                if class_name not in VALID_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                class_folder = output_path / class_name
                class_folder.mkdir(parents=True, exist_ok=True)
                
                h, w = img.shape[:2]
                pad = 3
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                
                filename = f"{img_file.stem}_{class_name}_{stats[class_name]:04d}.jpg"
                cv2.imwrite(str(class_folder / filename), crop)
                stats[class_name] += 1
    
    print("\n" + "="*50)
    print("Cropping Complete:")
    print("="*50)
    for cls, count in stats.items():
        print(f"  {cls}: {count}")
    print(f"  TOTAL: {sum(stats.values())}")
    
    with open(output_path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', '-y', default='detect_objects.pt')
    parser.add_argument('--source', '-s', default='data')
    parser.add_argument('--output', '-o', default='cropped')
    parser.add_argument('--device', '-d', default='cpu')
    parser.add_argument('--conf', '-c', type=float, default=0.4)
    args = parser.parse_args()
    
    crop_data(args.yolo, args.source, args.output, args.device, args.conf)
