import axios from 'axios';
import { FraudDetectionResult } from '@/types/fraud';

const API_ENDPOINT = process.env.NEXT_PUBLIC_API_ENDPOINT || 'http://localhost:5000/detect';

export async function detectFraud(imageBase64: string): Promise<FraudDetectionResult> {
  try {
    const base64Data = imageBase64.includes(',') 
      ? imageBase64.split(',')[1] 
      : imageBase64;

    const response = await axios.post<FraudDetectionResult>(
      API_ENDPOINT,
      { image: base64Data },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 120000,
      }
    );

    if (response.data.error) {
      throw new Error(response.data.error);
    }

    return response.data;
  } catch (error: unknown) {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        throw new Error(error.response.data?.error || 'Server error');
      } else if (error.request) {
        throw new Error('Cannot connect to server. Is the backend running?');
      }
    }
    throw error instanceof Error ? error : new Error('Unknown error');
  }
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
}
