// Detection result for each field
export interface Detection {
  field: string;
  yolo_conf?: number;
  recon_error: number;
  threshold: number;
  error_ratio: number;
  is_fraud: boolean;
  bbox: number[];
}

// API Response
export interface FraudDetectionResult {
  success: boolean;
  fraud_detected: boolean;
  fraud_reasons: string[];
  detections: Detection[];
  result_image?: string;
  error?: string;
}

// Component State
export interface UploadState {
  isLoading: boolean;
  error: string | null;
  result: FraudDetectionResult | null;
  previewImage: string | null;
}
