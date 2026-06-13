import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { FaSave, FaRedo, FaCheckCircle } from 'react-icons/fa';
import { switchModel, submitFeedback, getModels, MODEL_OPTIONS, TOTAL_ENDPOINTS } from '../services/api';

function Settings() {
  const [defaultModel, setDefaultModel] = useState('ensemble');
  const [availableModels, setAvailableModels] = useState([]);
  const [feedback, setFeedback] = useState({
    predictionId: '',
    correctLabel: '',
    confidence: '',
    notes: ''
  });
  const [apiUrl, setApiUrl] = useState('https://monocyclic-shara-unrotative.ngrok-free.dev');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [theme, setTheme] = useState('light');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadAvailableModels();
    loadSettings();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await getModels();
      setAvailableModels(response.data.models || []);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const loadSettings = () => {
    // Load from localStorage
    const savedSettings = localStorage.getItem('appSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      setDefaultModel(settings.defaultModel || 'ensemble');
      setApiUrl(settings.apiUrl || 'https://monocyclic-shara-unrotative.ngrok-free.dev');
      setAutoRefresh(settings.autoRefresh || false);
      setTheme(settings.theme || 'light');
    }
  };

  const handleSaveSettings = () => {
    const settings = {
      defaultModel,
      apiUrl,
      autoRefresh,
      theme
    };

    localStorage.setItem('appSettings', JSON.stringify(settings));
    toast.success('Settings saved successfully!');
  };

  const handleSwitchModel = async () => {
    setLoading(true);
    try {
      await switchModel(defaultModel);
      toast.success(`Default model switched to ${defaultModel}`);
    } catch (error) {
      console.error('Failed to switch model:', error);
      toast.error('Failed to switch model');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    if (!feedback.predictionId || !feedback.correctLabel) {
      toast.error('Please provide the prediction ID and correct diagnosis');
      return;
    }

    setLoading(true);
    try {
      const confidenceValue = feedback.confidence !== ''
        ? Number(feedback.confidence)
        : undefined;

      if (confidenceValue !== undefined && (Number.isNaN(confidenceValue) || confidenceValue < 0 || confidenceValue > 1)) {
        toast.error('Confidence must be a number between 0 and 1');
        setLoading(false);
        return;
      }

      await submitFeedback({
        predictionId: feedback.predictionId.trim(),
        correctLabel: feedback.correctLabel.trim(),
        confidence: confidenceValue,
        notes: feedback.notes.trim() || undefined,
      });

      toast.success('Feedback submitted successfully!');
      setFeedback({
        predictionId: '',
        correctLabel: '',
        confidence: '',
        notes: ''
      });
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      toast.error('Failed to submit feedback');
    } finally {
      setLoading(false);
    }
  };

  const handleResetSettings = () => {
    localStorage.removeItem('appSettings');
    setDefaultModel('ensemble');
    setApiUrl('https://monocyclic-shara-unrotative.ngrok-free.dev');
    setAutoRefresh(false);
    setTheme('light');
    toast.info('Settings reset to defaults');
  };

  const displayedAvailableModels = (availableModels.length ? availableModels : MODEL_OPTIONS).map((model) => {
    if (typeof model === 'string') {
      const match = MODEL_OPTIONS.find((option) => option.id === model || option.name === model);
      return {
        id: model,
        name: match?.name || model,
        description: match?.description,
      };
    }

    return {
      id: model.id || model.name,
      name: model.name || model.id,
      description: model.description,
    };
  });

  return (
    <div className="settings">
      <div className="panel-header">
        <h1>Settings</h1>
        <p>Configure application preferences and submit feedback</p>
      </div>

      <div className="settings-content">
        {/* Model Settings */}
        <div className="settings-section">
          <h2>Model Settings</h2>

          <div className="form-group">
            <label>Default Model</label>
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
            >
              {MODEL_OPTIONS.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} - {model.description}
                </option>
              ))}
            </select>
            <button
              className="btn-secondary"
              onClick={handleSwitchModel}
              disabled={loading}
            >
              Apply Model Switch
            </button>
          </div>

          {displayedAvailableModels.length > 0 && (
            <div className="available-models">
              <h4>Available Models ({displayedAvailableModels.length})</h4>
              <div className="models-list">
                {displayedAvailableModels.map((model) => (
                  <div key={model.id} className="model-item">
                    <FaCheckCircle className="check-icon" />
                    <span>
                      {model.name}
                      {model.description && (
                        <small className="model-description">{model.description}</small>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* API Configuration */}
        <div className="settings-section">
          <h2>API Configuration</h2>

          <div className="form-group">
            <label>Backend API URL</label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://monocyclic-shara-unrotative.ngrok-free.dev"
            />
            <small>Enter the base URL for the backend API</small>
          </div>

          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh system status
            </label>
          </div>
        </div>

        {/* Appearance Settings */}
        <div className="settings-section">
          <h2>Appearance</h2>

          <div className="form-group">
            <label>Theme</label>
            <select value={theme} onChange={(e) => setTheme(e.target.value)}>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto (System)</option>
            </select>
          </div>
        </div>

        {/* Feedback Form */}
        <div className="settings-section">
          <h2>Submit Feedback</h2>
          <p className="section-description">
            Help us improve the AI models by providing feedback on predictions
          </p>

          <div className="form-group">
            <label>Model Prediction</label>
            <input
              type="text"
              value={feedback.predictionId}
              onChange={(e) => setFeedback({ ...feedback, predictionId: e.target.value })}
              placeholder="Prediction ID from recent result"
            />
          </div>

          <div className="form-group">
            <label>Actual Diagnosis</label>
            <input
              type="text"
              value={feedback.correctLabel}
              onChange={(e) => setFeedback({ ...feedback, correctLabel: e.target.value })}
              placeholder="e.g., Pneumonia"
            />
          </div>

          <div className="form-group">
            <label>Confidence Level (0-1)</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={feedback.confidence}
              onChange={(e) => setFeedback({ ...feedback, confidence: e.target.value })}
              placeholder="e.g., 0.95"
            />
          </div>

          <div className="form-group">
            <label>Comments (Optional)</label>
            <textarea
              value={feedback.notes}
              onChange={(e) => setFeedback({ ...feedback, notes: e.target.value })}
              placeholder="Additional context or corrections..."
              rows={4}
            />
          </div>

          <button
            className="btn-primary"
            onClick={handleSubmitFeedback}
            disabled={loading}
          >
            Submit Feedback
          </button>
        </div>

        {/* Action Buttons */}
        <div className="settings-actions">
          <button className="btn-save" onClick={handleSaveSettings}>
            <FaSave /> Save Settings
          </button>
          <button className="btn-reset" onClick={handleResetSettings}>
            <FaRedo /> Reset to Defaults
          </button>
        </div>

        {/* System Information */}
        <div className="settings-section">
          <h2>System Information</h2>
          <div className="system-info-grid">
            <div className="info-row">
              <span className="info-label">Frontend Version:</span>
              <span className="info-value">1.0.0</span>
            </div>
            <div className="info-row">
              <span className="info-label">Total Endpoints:</span>
              <span className="info-value">{TOTAL_ENDPOINTS}</span>
            </div>
            <div className="info-row">
              <span className="info-label">Available Models:</span>
              <span className="info-value">{MODEL_OPTIONS.length}</span>
            </div>
            <div className="info-row">
              <span className="info-label">API Status:</span>
              <span className="info-value status-online">Connected</span>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="settings-section about-section">
          <h2>About</h2>
          <p>
            <strong>Chest X-Ray AI Analysis System</strong><br />
            Complete medical imaging analysis platform with {TOTAL_ENDPOINTS} API endpoints and {MODEL_OPTIONS.length} state-of-the-art AI models.
          </p>
          <p>
            Features include: Single & batch predictions, model comparison, GradCAM visualization,
            clinical reports, RAG-enhanced medical knowledge, and comprehensive analytics.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Settings;
