import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaUpload, FaSpinner } from 'react-icons/fa';
import { compareModels, MODEL_OPTIONS } from '../services/api';
import { validateImageLocally } from '../utils/validation';

const DEFAULT_COMPARISON_MODELS = MODEL_OPTIONS
  .filter((model) => model.id !== 'ensemble')
  .slice(0, 3)
  .map((model) => model.id);

function ModelComparison() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModels, setSelectedModels] = useState(DEFAULT_COMPARISON_MODELS);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

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

  const handleModelToggle = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(m => m !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const handleCompare = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    if (selectedModels.length < 2) {
      toast.error('Please select at least 2 models to compare');
      return;
    }

    setLoading(true);
    setComparisonResults(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      selectedModels.forEach(model => {
        formData.append('models', model);
      });

      const response = await compareModels(formData);
      setComparisonResults(response.data);
      console.log('Comparison Response:', response.data); // Debug log
      console.log('Results structure:', response.data.results); // Debug log
      toast.success('Comparison completed!');
    } catch (error) {
      console.error('Comparison failed:', error);
      console.error('Error response:', error.response?.data); // Debug log
      toast.error(error.response?.data?.detail || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const getBestModel = () => {
    if (!comparisonResults?.individual_results) return null;

    // Filter to only selected models
    const filteredResults = Object.entries(comparisonResults.individual_results)
      .filter(([modelName]) => {
        // Check if this model was in the selected models
        const modelId = MODEL_OPTIONS.find(m => m.name === modelName || m.id === modelName)?.id;
        return modelId && selectedModels.includes(modelId);
      });

    return filteredResults.reduce((best, [model, data]) => {
      if (!best || data.confidence > best.confidence) {
        return { model, ...data };
      }
      return best;
    }, null);
  };

  // Helper to get filtered results (only selected models)
  const getFilteredResults = () => {
    if (!comparisonResults?.individual_results) return {};

    const filtered = {};
    Object.entries(comparisonResults.individual_results).forEach(([modelName, data]) => {
      // Check if this model was in the selected models
      const modelId = MODEL_OPTIONS.find(m => m.name === modelName || m.id === modelName)?.id;
      if (modelId && selectedModels.includes(modelId)) {
        filtered[modelName] = data;
      }
    });
    return filtered;
  };

  return (
    <div className="model-comparison">
      <div className="panel-header">
        <h1>Model Comparison</h1>
        <p>Compare predictions across multiple AI models</p>
      </div>

      <div className="comparison-content">
        {/* Upload and Model Selection */}
        <div className="comparison-setup">
          <div className="upload-area" onClick={() => document.getElementById('compareFileInput').click()}>
            {preview ? (
              <img src={preview} alt="Preview" className="image-preview" />
            ) : (
              <div className="upload-placeholder">
                <FaUpload className="upload-icon" />
                <p>Click to upload chest X-ray</p>
              </div>
            )}
            <input
              id="compareFileInput"
              type="file"
              accept="image/*,.dcm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>

          <div className="model-selection">
            <h3>Select Models to Compare</h3>
            <div className="models-checklist">
              {MODEL_OPTIONS.map(model => (
                <label key={model.id} className="model-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelToggle(model.id)}
                  />
                  <span>{model.name}</span>
                  <small>{model.description}</small>
                </label>
              ))}
            </div>

            <button
              className="btn-compare"
              onClick={handleCompare}
              disabled={!selectedFile || selectedModels.length < 2 || loading}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinner" /> Comparing...
                </>
              ) : (
                `Compare ${selectedModels.length} Models`
              )}
            </button>
          </div>
        </div>

        {/* Comparison Results */}
        {comparisonResults && (
          <div className="comparison-results">
            <h2>📊 Model Comparison Analysis</h2>

            {/* Best Model Highlight */}
            {getBestModel() && (
              <div className="best-model-banner">
                <div className="trophy-icon">🏆</div>
                <div className="best-model-details">
                  <h3>Best Performing Model</h3>
                  <div className="best-model-name">{getBestModel().model}</div>
                  <div className="best-prediction-info">
                    <span className="best-prediction">{getBestModel().prediction}</span>
                    <span className="best-confidence">{(getBestModel().confidence * 100).toFixed(2)}% confidence</span>
                  </div>
                </div>
              </div>
            )}

            {/* Confidence Comparison Chart */}
            <div className="comparison-chart-section">
              <h3>🎯 Confidence Comparison</h3>
              <div className="horizontal-bar-chart">
                {Object.entries(getFilteredResults())
                  .sort((a, b) => b[1].confidence - a[1].confidence)
                  .map(([modelName, data]) => {
                    const isBest = getBestModel()?.model === modelName;
                    return (
                      <div key={modelName} className={`chart-row ${isBest ? 'best-row' : ''}`}>
                        <div className="chart-label">
                          {isBest && <span className="crown-icon">👑</span>}
                          <span className="model-name-label">{modelName}</span>
                        </div>
                        <div className="chart-bar-container">
                          <div
                            className="chart-bar"
                            style={{
                              width: `${(data.confidence * 100)}%`,
                              background: isBest
                                ? 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)'
                                : 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)'
                            }}
                          >
                            <span className="bar-value">{(data.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="chart-prediction">{data.prediction}</div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Detailed Model Cards */}
            <div className="detailed-models-section">
              <h3>🔬 Detailed Analysis by Model</h3>
              <div className="models-comparison-grid">
                {Object.entries(getFilteredResults()).map(([modelName, data]) => {
                  const isBest = getBestModel()?.model === modelName;
                  const diseaseColors = {
                    'COVID-19': '#ef4444',
                    'Normal': '#10b981',
                    'Pneumonia': '#f59e0b',
                    'Tuberculosis': '#8b5cf6'
                  };

                  return (
                    <div key={modelName} className={`detailed-model-card ${isBest ? 'best-card' : ''}`}>
                      {isBest && <div className="best-badge">⭐ Best Model</div>}

                      <h4 className="model-card-title">{modelName}</h4>

                      <div className="primary-prediction">
                        <div className="prediction-label-main">Prediction</div>
                        <div
                          className="prediction-value-main"
                          style={{ color: diseaseColors[data.prediction] || '#1f2937' }}
                        >
                          {data.prediction}
                        </div>
                        <div className="confidence-badge">
                          {(data.confidence * 100).toFixed(2)}% confident
                        </div>
                      </div>

                      {/* Probability Distribution */}
                      {data.all_probabilities && (
                        <div className="probability-distribution">
                          <div className="dist-header">Probability Distribution</div>
                          {Object.entries(data.all_probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .map(([disease, prob]) => (
                              <div key={disease} className="prob-bar-row">
                                <div className="prob-label">
                                  <span
                                    className="disease-dot"
                                    style={{ background: diseaseColors[disease] }}
                                  ></span>
                                  <span className="disease-name">{disease}</span>
                                </div>
                                <div className="prob-bar-container">
                                  <div
                                    className="prob-bar-fill"
                                    style={{
                                      width: `${(prob * 100)}%`,
                                      background: diseaseColors[disease]
                                    }}
                                  ></div>
                                </div>
                                <div className="prob-value">{(prob * 100).toFixed(1)}%</div>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Disease Probability Summary */}
            <div className="disease-summary-section">
              <h3>🦠 Disease Probability Summary Across All Models</h3>
              <div className="disease-bars-container">
                {(() => {
                  const diseases = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis'];
                  const diseaseColors = {
                    'COVID-19': '#ef4444',
                    'Normal': '#10b981',
                    'Pneumonia': '#f59e0b',
                    'Tuberculosis': '#8b5cf6'
                  };

                  return diseases.map(disease => {
                    const models = Object.entries(getFilteredResults());
                    const avgProb = models.reduce((sum, [_, data]) => {
                      return sum + (data.all_probabilities?.[disease] || 0);
                    }, 0) / (models.length || 1); // Avoid division by zero

                    const maxProb = Math.max(...models.map(([_, data]) =>
                      data.all_probabilities?.[disease] || 0
                    ));
                    const minProb = Math.min(...models.map(([_, data]) =>
                      data.all_probabilities?.[disease] || 0
                    ));

                    return (
                      <div key={disease} className="disease-summary-card">
                        <div className="disease-card-header">
                          <span
                            className="disease-dot-large"
                            style={{ background: diseaseColors[disease] }}
                          ></span>
                          <h4>{disease}</h4>
                        </div>
                        <div className="disease-stats">
                          <div className="stat-item">
                            <span className="stat-label">Average</span>
                            <span className="stat-value">{(avgProb * 100).toFixed(1)}%</span>
                          </div>
                          <div className="stat-item">
                            <span className="stat-label">Range</span>
                            <span className="stat-value">
                              {(minProb * 100).toFixed(1)}% - {(maxProb * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <div className="disease-avg-bar">
                          <div
                            className="disease-avg-fill"
                            style={{
                              width: `${(avgProb * 100)}%`,
                              background: diseaseColors[disease]
                            }}
                          ></div>
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>

            {/* Agreement Analysis */}
            {comparisonResults.agreement && (
              <div className="agreement-analysis">
                <h3>🤝 Model Agreement Analysis</h3>
                <div className="agreement-cards-grid">
                  <div className="agreement-metric-card">
                    <div className="metric-icon">
                      {comparisonResults.agreement.consensus ? '✅' : '⚠️'}
                    </div>
                    <div className="metric-label">Consensus</div>
                    <div className="metric-value">
                      {comparisonResults.agreement.consensus ? 'Full Agreement' : 'Mixed Predictions'}
                    </div>
                  </div>
                  <div className="agreement-metric-card">
                    <div className="metric-icon">📈</div>
                    <div className="metric-label">Agreement Rate</div>
                    <div className="metric-value">
                      {(comparisonResults.agreement.agreement_rate * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="agreement-metric-card">
                    <div className="metric-icon">🎯</div>
                    <div className="metric-label">Most Common</div>
                    <div className="metric-value">
                      {comparisonResults.agreement.most_common}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Processing Info */}
            <div className="comparison-footer">
              <div className="footer-stat">
                <strong>Models Compared:</strong> {Object.keys(getFilteredResults()).length}
              </div>
              {comparisonResults.processing_time && (
                <div className="footer-stat">
                  <strong>Processing Time:</strong> {comparisonResults.processing_time.toFixed(3)}s
                </div>
              )}
              {comparisonResults.timestamp && (
                <div className="footer-stat">
                  <strong>Timestamp:</strong> {new Date(comparisonResults.timestamp).toLocaleString()}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelComparison;
