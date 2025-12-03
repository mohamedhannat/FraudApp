'use client';

import { useState, useCallback } from 'react';
import { 
  Upload, Shield, AlertTriangle, CheckCircle, 
  Loader2, X, Image as ImageIcon 
} from 'lucide-react';
import { detectFraud, fileToBase64 } from '@/services/fraudApi';
import { FraudDetectionResult, UploadState, Detection } from '@/types/fraud';

export default function Home() {
  const [state, setState] = useState<UploadState>({
    isLoading: false,
    error: null,
    result: null,
    previewImage: null,
  });
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setState(s => ({ ...s, error: 'Please upload an image file' }));
      return;
    }

    setState({ isLoading: true, error: null, result: null, previewImage: null });

    try {
      const base64 = await fileToBase64(file);
      setState(s => ({ ...s, previewImage: base64 }));
      const result = await detectFraud(base64);
      setState(s => ({ ...s, isLoading: false, result }));
    } catch (err) {
      setState(s => ({
        ...s,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Detection failed',
      }));
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  }, [handleFile]);

  const reset = () => setState({ isLoading: false, error: null, result: null, previewImage: null });

  const getBarColor = (ratio: number) => 
    ratio > 1 ? 'bg-red-500' : ratio > 0.7 ? 'bg-yellow-500' : 'bg-green-500';
  
  const getTextColor = (ratio: number) => 
    ratio > 1 ? 'text-red-400' : ratio > 0.7 ? 'text-yellow-400' : 'text-green-400';

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
          <Shield className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-xl font-bold text-white">ID Card Fraud Detection</h1>
            <p className="text-sm text-slate-400">Anomaly Detection System</p>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload */}
          <div className="space-y-6">
            <div
              className={`relative border-2 border-dashed rounded-xl p-8 transition-all ${
                dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 bg-slate-800/50 hover:border-slate-500'
              }`}
              onDragOver={e => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={state.isLoading}
              />
              <div className="text-center">
                {state.isLoading ? (
                  <Loader2 className="w-12 h-12 text-blue-500 mx-auto animate-spin" />
                ) : (
                  <Upload className="w-12 h-12 text-slate-400 mx-auto" />
                )}
                <h3 className="mt-4 text-lg font-medium text-white">
                  {state.isLoading ? 'Analyzing...' : 'Upload ID Card Image'}
                </h3>
                <p className="mt-2 text-sm text-slate-400">
                  {state.isLoading ? 'Running anomaly detection' : 'Drag & drop or click to browse'}
                </p>
              </div>
            </div>

            {state.previewImage && (
              <div className="relative rounded-xl overflow-hidden bg-slate-800">
                <button onClick={reset} className="absolute top-2 right-2 p-1 bg-slate-900/80 rounded-full hover:bg-slate-700 z-10">
                  <X className="w-5 h-5 text-white" />
                </button>
                <img src={state.previewImage} alt="Preview" className="w-full" />
              </div>
            )}

            {state.error && (
              <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-xl">
                <div className="flex items-center gap-2 text-red-400">
                  <AlertTriangle className="w-5 h-5" />
                  <span className="font-medium">Error</span>
                </div>
                <p className="mt-1 text-sm text-red-300">{state.error}</p>
              </div>
            )}
          </div>

          {/* Results */}
          <div className="space-y-6">
            {state.result ? (
              <>
                {/* Verdict */}
                <div className={`p-6 rounded-xl border ${
                  state.result.fraud_detected ? 'bg-red-500/10 border-red-500/50' : 'bg-green-500/10 border-green-500/50'
                }`}>
                  <div className="flex items-center gap-3">
                    {state.result.fraud_detected ? (
                      <AlertTriangle className="w-10 h-10 text-red-500" />
                    ) : (
                      <CheckCircle className="w-10 h-10 text-green-500" />
                    )}
                    <div>
                      <h2 className={`text-2xl font-bold ${state.result.fraud_detected ? 'text-red-400' : 'text-green-400'}`}>
                        {state.result.fraud_detected ? 'FRAUD DETECTED' : 'AUTHENTIC'}
                      </h2>
                      <p className="text-sm text-slate-400">
                        {state.result.fraud_detected ? 'Document shows signs of manipulation' : 'No anomalies detected'}
                      </p>
                    </div>
                  </div>

                  {state.result.fraud_reasons?.length > 0 && (
                    <div className="mt-4 space-y-1">
                      <h4 className="text-sm font-medium text-red-400">Issues:</h4>
                      {state.result.fraud_reasons.map((r, i) => (
                        <div key={i} className="text-sm text-red-300">• {r}</div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Field Analysis */}
                {state.result.detections?.length > 0 && (
                  <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                    <h3 className="font-medium text-white mb-4">Field Analysis</h3>
                    <div className="space-y-4">
                      {state.result.detections.map((det: Detection, i: number) => (
                        <div key={i} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              {det.is_fraud ? (
                                <AlertTriangle className="w-4 h-4 text-red-400" />
                              ) : (
                                <CheckCircle className="w-4 h-4 text-green-400" />
                              )}
                              <span className="text-white font-medium capitalize">{det.field}</span>
                            </div>
                            <span className={`text-sm font-mono ${getTextColor(det.error_ratio)}`}>
                              {det.error_ratio.toFixed(2)}x
                            </span>
                          </div>
                          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${getBarColor(det.error_ratio)} transition-all`}
                              style={{ width: `${Math.min(det.error_ratio * 100, 100)}%` }}
                            />
                          </div>
                          <div className="flex justify-between text-xs text-slate-500">
                            <span>Error: {det.recon_error.toFixed(6)}</span>
                            <span>Threshold: {det.threshold.toFixed(6)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-700 text-xs text-slate-500">
                      <span className="text-green-400">●</span> &lt;0.7x Normal | 
                      <span className="text-yellow-400 ml-2">●</span> 0.7-1x Warning | 
                      <span className="text-red-400 ml-2">●</span> &gt;1x Fraud
                    </div>
                  </div>
                )}

                {/* Result Image */}
                {state.result.result_image && (
                  <div className="rounded-xl overflow-hidden bg-slate-800 border border-slate-700">
                    <div className="p-3 border-b border-slate-700 flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-white" />
                      <h3 className="font-medium text-white">Annotated Result</h3>
                    </div>
                    <img src={`data:image/jpeg;base64,${state.result.result_image}`} alt="Result" className="w-full" />
                  </div>
                )}
              </>
            ) : (
              <div className="h-full flex items-center justify-center p-8 bg-slate-800/50 rounded-xl border border-slate-700">
                <div className="text-center text-slate-400">
                  <Shield className="w-16 h-16 mx-auto opacity-50" />
                  <p className="mt-4">Upload an ID card to begin</p>
                  <p className="mt-2 text-sm text-slate-500">Analyzes: firstName, lastName, photo</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
