import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaUpload, FaSpinner, FaDownload } from 'react-icons/fa';
import {
  generateGradCAM,
  enhancedGradCAM,
  gradcamHeatmap,
  compareGradCAM,
  compareGradCAMBase64,
  explainPrediction,
  MODEL_OPTIONS
} from '../services/api';
import { validateImageLocally } from '../utils/validation';

function GradCAMViewer() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('DenseNet169');
  const [gradcamType, setGradcamType] = useState('standard');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate before accepting
    toast.info('Analyzing image with AI...');
    const validation = await validateImageLocally(file);
    if (!validation.is_valid) {
      toast.error(validation.message || 'Not a valid chest X-ray. Please upload a medical X-ray image.');
      e.target.value = '';
      return;
    }

    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleGenerateGradCAM = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', selectedModel);

      // Special handling for ViT and EfficientNet models
      const isViTModel = selectedModel.includes('ViT');
      const isEfficientNet = selectedModel.includes('EfficientNet');

      if (isViTModel || isEfficientNet) {
        toast.info(`Generating GradCAM for ${selectedModel}... This may take longer.`, { autoClose: 3000 });
      }

      let response;
      switch (gradcamType) {
        case 'enhanced':
          response = await enhancedGradCAM(formData);
          break;
        case 'heatmap':
          response = await gradcamHeatmap(formData);
          break;
        case 'explain':
          response = await explainPrediction(formData);
          break;
        case 'compare':
          response = await compareGradCAM(formData);
          break;
        case 'compare_base64':
          response = await compareGradCAMBase64(formData);
          break;
        default:
          response = await generateGradCAM(formData);
      }

      console.log('GradCAM Response:', response.data);
      console.log('Available fields:', Object.keys(response.data));

      // Validate response has visualization data
      const hasVisualization = response.data.gradcam ||
        response.data.heatmap ||
        response.data.overlay ||
        response.data.attention_map ||
        response.data.gradcam_images;

      if (!hasVisualization) {
        console.warn('No visualization data in response:', response.data);
        toast.warning('GradCAM generated but no visualization available. Check console for details.');
      } else {
        toast.success('GradCAM generated successfully!');
      }

      setResult(response.data);
    } catch (error) {
      console.error('GradCAM generation failed:', error);
      console.error('Error response:', error.response?.data);

      const errorMsg = error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'GradCAM generation failed';

      toast.error(errorMsg);

      // Show specific error for ViT/EfficientNet if applicable
      if ((selectedModel.includes('ViT') || selectedModel.includes('EfficientNet')) &&
        errorMsg.includes('not supported')) {
        toast.info('Try using "Enhanced GradCAM" or "Heatmap" for transformer models', { autoClose: 5000 });
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadVisualization = (imageData, filename) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imageData}`;
    link.download = filename;
    link.click();
    toast.success('Visualization downloaded');
  };

  return (
    <div className="gradcam-viewer">
      <div className="panel-header">
        <h1>GradCAM Visualization</h1>
        <p>Explainable AI - See what the model is looking at</p>
      </div>

      <div className="gradcam-content">
        {/* Info Banner for Model-Specific Guidance */}
        {selectedModel && (selectedModel.includes('ViT') || selectedModel.includes('EfficientNet')) && (
          <div className="gradcam-info-banner">
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <p>
              <strong>{selectedModel}</strong> is a {selectedModel.includes('ViT') ? 'Vision Transformer' : 'EfficientNet'} model.
              For best results, use <strong>"Enhanced GradCAM"</strong> or <strong>"Heatmap"</strong> visualization types.
            </p>
          </div>
        )}

        {/* Upload and Configuration */}
        <div className="gradcam-setup">
          <div className="upload-area" onClick={() => document.getElementById('gradcamFileInput').click()}>
            <input
              id="gradcamFileInput"
              type="file"
              accept="image/*,.dcm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            {preview ? (
              <div className="image-preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <p style={{ marginTop: '12px', color: '#6b7280', fontSize: '14px' }}>
                  {selectedFile?.name}
                </p>
              </div>
            ) : (
              <div className="upload-placeholder">
                <FaUpload size={48} style={{ marginBottom: '16px' }} />
                <p style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
                  Click to upload chest X-ray
                </p>
                <p style={{ fontSize: '14px', color: '#6b7280' }}>
                  Supports: JPG, PNG, DICOM (.dcm)
                </p>
              </div>
            )}
          </div>

          <div className="model-selection">
            <label>Select Model</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {MODEL_OPTIONS.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          <div className="gradcam-type-selection">
            <label>GradCAM Visualization Type</label>
            <select value={gradcamType} onChange={(e) => setGradcamType(e.target.value)}>
              <option value="standard">🔍 Standard GradCAM</option>
              <option value="enhanced">✨ Enhanced GradCAM</option>
              <option value="heatmap">🔥 Heatmap Only</option>
              <option value="explain">📊 Full Explanation</option>
              <option value="compare">🔄 Compare Models</option>
              <option value="compare_base64">📸 Compare (Base64)</option>
            </select>
          </div>

          <button
            className="btn-primary btn-generate"
            onClick={handleGenerateGradCAM}
            disabled={!selectedFile || loading}
            style={{ marginTop: '24px', width: '100%', padding: '16px', fontSize: '16px', fontWeight: '600' }}
          >
            {loading ? (
              <>
                <FaSpinner className="spinner" /> Generating...
              </>
            ) : (
              <>
                <FaDownload /> Generate GradCAM
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div className="gradcam-results">
            <h2>Visualization Results</h2>

            {/* Prediction Info */}
            <div className="prediction-info-card">
              <h3>Prediction: {result.prediction}</h3>
              <p className="confidence">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
              {result.model_used && <p className="model-info">Model: {result.model_used}</p>}
            </div>

            {/* Visualizations Grid */}
            <div className="visualizations-grid">
              {/* Standard GradCAM - from standard endpoint */}
              {result.gradcam && !result.heatmap && !result.overlay && (
                <div className="viz-card">
                  <h4>GradCAM Visualization</h4>
                  <img
                    src={result.gradcam.startsWith('data:') ? result.gradcam : `data:image/png;base64,${result.gradcam}`}
                    alt="GradCAM"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.gradcam.replace('data:image/png;base64,', ''),
                      'gradcam.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}

              {/* Heatmap - from heatmap endpoint */}
              {result.heatmap && (
                <div className="viz-card">
                  <h4>Heatmap</h4>
                  <img
                    src={result.heatmap.startsWith('data:') ? result.heatmap : `data:image/png;base64,${result.heatmap}`}
                    alt="GradCAM Heatmap"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.heatmap.replace('data:image/png;base64,', ''),
                      'heatmap.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}

              {/* Overlay - from enhanced endpoint */}
              {result.overlay && (
                <div className="viz-card">
                  <h4>Overlay</h4>
                  <img
                    src={result.overlay.startsWith('data:') ? result.overlay : `data:image/png;base64,${result.overlay}`}
                    alt="GradCAM Overlay"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.overlay.replace('data:image/png;base64,', ''),
                      'overlay.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}

              {/* Attention Map - from explain endpoint */}
              {result.attention_map && (
                <div className="viz-card">
                  <h4>Attention Map</h4>
                  <img
                    src={result.attention_map.startsWith('data:') ? result.attention_map : `data:image/png;base64,${result.attention_map}`}
                    alt="Attention Map"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.attention_map.replace('data:image/png;base64,', ''),
                      'attention.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}

              {/* Token Importance - from explain endpoint */}
              {result.token_importance && (
                <div className="viz-card">
                  <h4>Token Importance</h4>
                  <img
                    src={result.token_importance.startsWith('data:') ? result.token_importance : `data:image/png;base64,${result.token_importance}`}
                    alt="Token Importance"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.token_importance.replace('data:image/png;base64,', ''),
                      'tokens.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}

              {/* Integrated Gradients - from explain endpoint */}
              {result.integrated_gradients && (
                <div className="viz-card">
                  <h4>Integrated Gradients</h4>
                  <img
                    src={result.integrated_gradients.startsWith('data:') ? result.integrated_gradients : `data:image/png;base64,${result.integrated_gradients}`}
                    alt="Integrated Gradients"
                    className="gradcam-image"
                  />
                  <button
                    className="btn-download-small"
                    onClick={() => downloadVisualization(
                      result.integrated_gradients.replace('data:image/png;base64,', ''),
                      'gradients.png'
                    )}
                  >
                    <FaDownload /> Download
                  </button>
                </div>
              )}
            </div>

            {/* Multiple Model Comparisons - from compare endpoints */}
            {result.gradcam_images && Object.keys(result.gradcam_images).length > 0 && (
              <div className="model-gradcams">
                <h3>Model Comparison</h3>
                <div className="visualizations-grid">
                  {Object.entries(result.gradcam_images).map(([modelName, imageData]) => (
                    <div key={modelName} className="viz-card">
                      <h4>{modelName}</h4>
                      <img
                        src={imageData.startsWith('data:') ? imageData : `data:image/png;base64,${imageData}`}
                        alt={`${modelName} GradCAM`}
                        className="gradcam-image"
                      />
                      <button
                        className="btn-download-small"
                        onClick={() => downloadVisualization(
                          imageData.replace('data:image/png;base64,', ''),
                          `${modelName}_gradcam.png`
                        )}
                      >
                        <FaDownload /> Download
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Patches */}
            {result.top_patches && (
              <div className="patches-section">
                <h3>Most Important Image Patches</h3>
                <div className="patches-list">
                  {result.top_patches.map((patch, index) => (
                    <div key={index} className="patch-item">
                      <span className="patch-rank">#{index + 1}</span>
                      <span className="patch-value">Patch {patch}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Important Regions */}
            {result.important_regions && (
              <div className="regions-section">
                <h3>Important Regions Detected</h3>
                <div className="regions-list">
                  {result.important_regions.map((region, index) => (
                    <div key={index} className="region-item">
                      <p>Region {index + 1}</p>
                      <div className="region-coords">
                        {Object.entries(region).map(([key, value]) => (
                          <span key={key}>{key}: {value}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Text Explanation - from explain endpoint */}
            {result.explanation && (
              <div className="explanation-section">
                <h3>AI Explanation</h3>
                <div className="explanation-text">
                  <p>{result.explanation}</p>
                </div>
              </div>
            )}

            {/* Feature Importance - from explain endpoint */}
            {result.feature_importance && (
              <div className="feature-importance-section">
                <h3>Feature Importance</h3>
                <div className="feature-list">
                  {Object.entries(result.feature_importance).map(([feature, importance]) => (
                    <div key={feature} className="feature-item">
                      <span className="feature-name">{feature}</span>
                      <div className="importance-bar">
                        <div
                          className="importance-fill"
                          style={{ width: `${importance * 100}%` }}
                        ></div>
                      </div>
                      <span className="importance-value">{(importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Fusion Weights */}
            {result.model_fusion_weights && (
              <div className="fusion-weights">
                <h3>Model Fusion Weights</h3>
                <div className="weights-bars">
                  {Object.entries(result.model_fusion_weights).map(([model, weight]) => (
                    <div key={model} className="weight-bar">
                      <span className="weight-label">{model}</span>
                      <div className="bar-container">
                        <div
                          className="bar-fill"
                          style={{ width: `${weight * 100}%` }}
                        ></div>
                      </div>
                      <span className="weight-value">{(weight * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Probabilities */}
            {result.probabilities && (
              <div className="probs-section">
                <h3>Disease Probabilities</h3>
                <div className="probs-chart">
                  {Object.entries(result.probabilities).map(([disease, prob]) => (
                    <div key={disease} className="prob-bar-item">
                      <span className="prob-label">{disease}</span>
                      <div className="prob-bar-container">
                        <div
                          className="prob-bar-fill"
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                      <span className="prob-value">{(prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Processing Info */}
            <div className="gradcam-meta">
              {result.processing_time && (
                <p><strong>Processing Time:</strong> {result.processing_time.toFixed(3)}s</p>
              )}
              {result.timestamp && (
                <p><strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default GradCAMViewer;
