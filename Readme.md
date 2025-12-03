# ID Card Fraud Detection System

A complete local fraud detection system using deep learning (YOLO + Autoencoder) to detect manipulated ID cards.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This system detects fraud in Egyptian ID cards by analyzing three key fields:
- **firstName** - First name field
- **lastName** - Last name field  
- **photo** - Photo field

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. YOLO Detection                                          │
│     Detects and crops: firstName, lastName, photo           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Autoencoder Analysis                                    │
│     Trained on VALID samples only                           │
│     Learns what "normal" fields look like                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Anomaly Detection                                       │
│     Low reconstruction error  →  AUTHENTIC ✅               │
│     High reconstruction error →  FRAUD 🚨                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- YOLO model file (`detect_objects.pt`)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
## 📁 Project Structure

```

├── fraud-detection-backend/  # Python Backend
│   ├── requirements.txt
│   ├── crop_data.py         # Step 1: Crop training data
│   ├── train.py             # Step 2: Train models
│   ├── predict.py           # Step 3: Test/predict
│   ├── server.py            # Step 4: API server
│   ├── data/                # Put ID card images here
│   ├── cropped/             # Cropped training data
│   ├── models/              # Trained models
│   └── results/             # Prediction results
│
└── fraud-detection-frontend/ # Next.js Frontend
    ├── package.json
    ├── src/
    │   ├── app/page.tsx     # Main UI
    │   ├── services/        # API calls
    │   └── types/           # TypeScript types
    └── .env.local           # API endpoint config
```

## 🔧 Backend Usage

### Step 1: Install Dependencies

```bash
cd fraud-detection-backend
pip install -r requirements.txt
```

### Step 2: Prepare Data

1. Copy your YOLO model (`detect_objects.pt`) to the backend folder
2. Add ID card images to the `data/` folder

### Step 3: Crop Training Data

```bash
python crop_data.py --yolo detect_objects.pt --source data/ --output cropped/
```

**Output:**
```
Cropping Complete:
  firstName: 337 samples
  lastName: 363 samples
  photo: 287 samples
```

### Step 4: Train Models

```bash
python train.py --data cropped/ --output models/ --epochs 300
```

**Output:**
```
Training: firstName
  Loaded 337 images
  Epoch [30/300] Loss: 0.012345
  ...
  Threshold: 0.001286

Training: lastName
  ...

TRAINING COMPLETE
```

### Step 5: Test Prediction

```bash
# Single image
python predict.py --source test.jpg

# Folder of images
python predict.py --source test_images/
```

**Output:**
```
FRAUD DETECTION RESULT
══════════════════════════════════════════════════════════════
🚨 VERDICT: FRAUD DETECTED!

Field Analysis:
  firstName: ✅ OK
    Error: 0.001200 | Threshold: 0.001286
    Ratio: 0.93x

  photo: ❌ FRAUD
    Error: 0.003500 | Threshold: 0.001897
    Ratio: 1.84x

Result saved: results/result_test.jpg
```

### Step 6: Start API Server

```bash
python server.py --port 5000
```

**Server running at:** http://localhost:5000

## 🌐 Frontend Usage

### Install & Run

```bash
cd fraud-detection-frontend
npm install
npm run dev
```

**Open:** http://localhost:3000

### Configuration

Edit `.env.local` to set API endpoint:

```env
NEXT_PUBLIC_API_ENDPOINT=http://localhost:5000/detect
```

## 📡 API Reference

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models": ["firstName", "lastName", "photo"]
}
```

### Detect Fraud

```bash
POST /detect
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "success": true,
  "fraud_detected": true,
  "fraud_reasons": [
    "photo: 1.84x threshold"
  ],
  "detections": [
    {
      "field": "firstName",
      "recon_error": 0.001200,
      "threshold": 0.001286,
      "error_ratio": 0.93,
      "is_fraud": false
    },
    {
      "field": "photo",
      "recon_error": 0.003500,
      "threshold": 0.001897,
      "error_ratio": 1.84,
      "is_fraud": true
    }
  ],
  "result_image": "base64_encoded_annotated_image"
}
```

### Test with cURL

```bash
# Health check
curl http://localhost:5000/health

# Detect fraud
BASE64=$(base64 -w 0 test.jpg)  # Linux
# BASE64=$(base64 -i test.jpg)  # macOS
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$BASE64\"}"
```

## ⚙️ Configuration

### Adjust Sensitivity

Edit `models/<field>/config.json`:

```json
{
  "threshold": 0.001286  // Lower = more sensitive
}
```

### GPU Support

```bash
# Train with GPU
python train.py --device cuda

# Server with GPU
python server.py --device cuda
```

## 🎯 Detection Logic

| Error Ratio | Status | Meaning |
|-------------|--------|---------|
| < 0.7x | 🟢 Normal | Field looks authentic |
| 0.7x - 1.0x | 🟡 Warning | Slight anomaly |
| > 1.0x | 🔴 Fraud | Field appears manipulated |

## 📊 Model Architecture

### Autoencoder

```
Input (224x224x3)
    │
    ▼
Encoder (Conv2d + InstanceNorm + LeakyReLU)
    │ 224 → 112 → 56 → 28 → 14 → 7
    ▼
Latent Space (128-dim)
    │
    ▼
Decoder (ConvTranspose2d + InstanceNorm + ReLU)
    │ 7 → 14 → 28 → 56 → 112 → 224
    ▼
Output (224x224x3)
```

## 🐛 Troubleshooting

### "No images found"
- Add ID card images to `data/` folder
- Supported formats: jpg, jpeg, png

### "Cannot connect to server"
- Make sure backend is running: `python server.py`
- Check port 5000 is not in use

### "CUDA out of memory"
- Use CPU: `--device cpu`
- Reduce batch size: `--batch-size 8`

### Low detection accuracy
- Add more training images
- Increase epochs: `--epochs 500`
- Lower threshold in config.json

## 📝 License

MIT License - feel free to use for any purpose.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open a Pull Request

## 📧 Contact

For questions or support, open an issue on GitHub.

---

Made with ❤️ for secure document verification
