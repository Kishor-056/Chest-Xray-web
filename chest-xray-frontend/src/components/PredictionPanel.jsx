import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaUpload, FaSpinner, FaDownload } from 'react-icons/fa';
import {
  predict,
  predictRealtime,
  predictStream,
  analyzeWithAgent,
  validateXray,
  generateGradCAM,
  MODEL_OPTIONS
} from '../services/api';
import { validateImageLocally } from '../utils/validation';
import { generateXAIReport } from '../utils/xaiEngine';
import XAIReportCard from './XAIReportCard';

function PredictionPanel() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [predictionType, setPredictionType] = useState('standard');
  const [result, setResult] = useState(null);
  const [xaiReport, setXaiReport] = useState(null);
  const [streamStatus, setStreamStatus] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);
    setResult(null);
    setXaiReport(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', selectedModel);

      // VALIDATION STEP (Client-Side First — uses MobileNet ML model)
      toast.info('Analyzing image with AI — may take a moment on first run...');
      const localValidation = await validateImageLocally(selectedFile);

      if (!localValidation.is_valid) {
        toast.error(localValidation.message || 'Image validation failed.');
        setLoading(false);
        return;
      }

      // Backend Validation (Optional - try/catch to handle 404)
      try {
        const validationResponse = await validateXray(formData);
        if (validationResponse.data && !validationResponse.data.is_valid && validationResponse.data.is_valid !== undefined) {
          toast.error(validationResponse.data.message || 'Server: Image is not a valid chest X-ray');
          setLoading(false);
          return;
        }
      } catch (err) {
        // Ignore 404 from backend validation, as we already validated locally
        if (err.response && err.response.status === 404) {
          console.warn("Backend validation endpoint not found (404), proceeding with local validation only.");
        } else {
          console.warn("Backend validation error:", err);
          // Optional: fail on other errors? For now, proceed if it's just a server issue, 
          // but maybe we should warn.
        }
      }

      let data;
      let streamUpdates = [];
      setStreamStatus([]);
      switch (predictionType) {
        case 'realtime':
          ({ data } = await predictRealtime(formData));
          break;
        case 'stream':
          const streamResult = await predictStream(formData, (message) => {
            streamUpdates = [...streamUpdates, message];
            setStreamStatus(prev => [...prev, message]);
          });
          if (streamResult?.result) {
            data = streamResult.result;
          } else {
            data = streamResult;
          }
          break;
        case 'analyze':
          ({ data } = await analyzeWithAgent(formData));
          break;
        default:
          ({ data } = await predict(formData));
      }

      const normalized = normalizePredictionResult(data, {
        selectedModel,
        type: predictionType,
        streamUpdates,
      });

      setResult(normalized);

      // ── Generate Advanced XAI Report ────────────────────────────────────
      try {
        // Fetch GradCAM heatmap from backend (non-blocking)
        let gradcamUrl = null;
        try {
          const gcFormData = new FormData();
          gcFormData.append('file', selectedFile);
          gcFormData.append('model_name', selectedModel);
          const gcRes = await generateGradCAM(gcFormData);
          if (gcRes.data?.gradcam_image) {
            gradcamUrl = `data:image/png;base64,${gcRes.data.gradcam_image}`;
          } else if (gcRes.data?.heatmap_base64) {
            gradcamUrl = `data:image/png;base64,${gcRes.data.heatmap_base64}`;
          }
        } catch (gcErr) {
          console.warn('GradCAM not available:', gcErr.message);
        }

        const prediction = normalized.prediction;
        const confidence = normalized.confidence;
        const allProbs = normalized.all_probabilities;
        const report = generateXAIReport(prediction, confidence, allProbs, gradcamUrl);
        setXaiReport(report);
      } catch (xaiErr) {
        console.warn('XAI report generation failed:', xaiErr);
      }

      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction failed:', error);
      toast.error(error.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const normalizePredictionResult = (data, { selectedModel, type, streamUpdates = [] }) => {
    if (!data) {
      return null;
    }

    const basePrediction = data.prediction || data.diagnosis || 'Unknown';
    const probabilities = data.all_probabilities || data.probabilities;
    const confidence = typeof data.confidence === 'number'
      ? data.confidence
      : (typeof data?.result?.confidence === 'number' ? data.result.confidence : null);

    return {
      ...data,
      prediction: basePrediction,
      confidence: confidence ?? 0,
      all_probabilities: probabilities,
      model_used: data.model_used
        || (selectedModel === 'ensemble' ? 'Ensemble' : selectedModel),
      timestamp: data.timestamp || new Date().toISOString(),
      rag_context: data.rag_context,
      processing_updates: type === 'stream' ? streamUpdates : undefined,
    };
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.9) return '#10b981';
    if (confidence > 0.7) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="prediction-panel">
      <div className="panel-header">
        <h1>Single Image Prediction</h1>
        <p>Upload a chest X-ray for AI analysis</p>
      </div>

      <div className="prediction-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div className="upload-area" onClick={() => document.getElementById('fileInput').click()}>
            {preview ? (
              <img src={preview} alt="Preview" className="image-preview" />
            ) : (
              <div className="upload-placeholder">
                <FaUpload className="upload-icon" />
                <p>Click to upload chest X-ray</p>
                <p className="upload-hint">PNG, JPG, JPEG, DICOM supported</p>
              </div>
            )}
            <input
              id="fileInput"
              type="file"
              accept="image/*,.dcm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>

          {/* Configuration */}
          <div className="config-section">
            <div className="form-group">
              <label>Select Model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {MODEL_OPTIONS.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.description}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Prediction Type</label>
              <select value={predictionType} onChange={(e) => setPredictionType(e.target.value)}>
                <option value="standard">Standard Prediction</option>
                <option value="realtime">Real-time (Fast)</option>
                <option value="stream">Streaming</option>
                <option value="analyze">AI Agent Analysis</option>
              </select>
            </div>

            <button
              className="btn-predict"
              onClick={handlePredict}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinner" /> Processing...
                </>
              ) : (
                'Predict Disease'
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className="results-section">
            <h2>Prediction Results</h2>
            {streamStatus.length > 0 && (
              <div className="result-card">
                <h3>Streaming Updates</h3>
                <ul className="stream-updates">
                  {streamStatus.map((message, index) => (
                    <li key={index}>
                      <span className="update-status">{message.status || 'update'}:</span>
                      <span className="update-payload">{JSON.stringify(message)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Main Prediction */}
            <div className="result-card main-result">
              <h3>Diagnosis</h3>
              <div className="diagnosis-box">
                <p className="prediction-label">{result.prediction}</p>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${(result.confidence * 100)}%`,
                      backgroundColor: getConfidenceColor(result.confidence)
                    }}
                  ></div>
                </div>
                <p className="confidence-text">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            {/* All Probabilities */}
            {result.all_probabilities && (
              <div className="result-card">
                <h3>All Disease Probabilities</h3>
                <div className="probabilities-list">
                  {Object.entries(result.all_probabilities).map(([disease, prob]) => (
                    <div key={disease} className="probability-item">
                      <span className="disease-name">{disease}</span>
                      <div className="prob-bar-container">
                        <div
                          className="prob-bar"
                          style={{ width: `${(prob * 100)}%` }}
                        ></div>
                      </div>
                      <span className="prob-value">{(prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* AI Agent Reasoning Steps */}
            {result.reasoning_steps && (
              <div className="result-card">
                <h3>AI Agent Reasoning</h3>
                <div className="reasoning-steps">
                  {result.reasoning_steps.map((step, index) => (
                    <div key={index} className="reasoning-step">
                      <span className="step-number">{index + 1}</span>
                      <p>{step}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medical Knowledge */}
            {result.medical_knowledge && (
              <div className="result-card">
                <h3>Related Medical Knowledge</h3>
                <div className="knowledge-list">
                  {result.medical_knowledge.map((item, index) => (
                    <div key={index} className="knowledge-item">
                      <p className="knowledge-text">{item.text}</p>
                      <span className="knowledge-tag">{item.condition}</span>
                      <span className="relevance-score">
                        Relevance: {(item.relevance_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Explanation */}
            {result.explanation && (
              <div className="result-card">
                <h3>Clinical Explanation</h3>
                <p className="explanation-text">{result.explanation}</p>
              </div>
            )}

            {result.rag_context && (
              <div className="result-card">
                <h3>Retrieved Medical Context</h3>
                <p className="rag-context">{result.rag_context}</p>
              </div>
            )}

            {/* Additional Info */}
            <div className="result-meta">
              <p><strong>Model Used:</strong> {result.model_used}</p>
              {result.inference_time_seconds && (
                <p><strong>Inference Time:</strong> {result.inference_time_seconds.toFixed(3)}s</p>
              )}
              {result.timestamp && (
                <p><strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}</p>
              )}
            </div>
          </div>
        )}

        {/* ── Advanced XAI Report Card ── */}
        {xaiReport && (
          <div className="xai-report-wrapper">
            <h2 style={{ color: '#e0e0e0', marginBottom: 12, fontSize: 18 }}>🧠 Explainable AI — Clinical Analysis Report</h2>
            <XAIReportCard report={xaiReport} />
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionPanel;
