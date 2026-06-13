import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { FaCheckCircle, FaExclamationTriangle, FaRobot, FaDatabase } from 'react-icons/fa';
import { detailedHealthCheck, getModels, getMetrics, MODEL_OPTIONS, TOTAL_ENDPOINTS, DISEASE_CLASSES } from '../services/api';
import { useNavigate } from 'react-router-dom';

function Dashboard() {
  const [healthData, setHealthData] = useState(null);
  const [models, setModels] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [healthRes, modelsRes, metricsRes] = await Promise.all([
        detailedHealthCheck(),
        getModels(),
        getMetrics()
      ]);

      setHealthData(healthRes.data);
      setModels(modelsRes.data.models || []);
      setMetrics(metricsRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load dashboard:', error);
      toast.error('Failed to load dashboard data');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  const displayedModels = (models.length ? models : MODEL_OPTIONS).map((model) => {
    if (typeof model === 'string') {
      const match = MODEL_OPTIONS.find((option) => option.id === model || option.name === model);
      return {
        id: model,
        name: match?.name || model,
        description: match?.description || 'Active model',
      };
    }

    return {
      id: model.id || model.name,
      name: model.name || model.id,
      description: model.description || 'Active model',
    };
  });

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Medical AI Dashboard</h1>
        <p>
          Complete Chest X-Ray Analysis System with {TOTAL_ENDPOINTS} Endpoints & {MODEL_OPTIONS.length} Models
        </p>
      </div>

      {/* System Status Cards */}
      <div className="cards-grid">
        <div className="card card-success">
          <FaCheckCircle className="card-icon" />
          <div className="card-content">
            <h3>System Status</h3>
            <p className="card-value">{healthData?.status || 'Unknown'}</p>
            <p className="card-label">All systems operational</p>
          </div>
        </div>

        <div className="card card-info">
          <FaRobot className="card-icon" />
          <div className="card-content">
            <h3>Models Loaded</h3>
            <p className="card-value">{healthData?.system?.models_loaded || MODEL_OPTIONS.length}</p>
            <p className="card-label">AI models ready</p>
          </div>
        </div>

        <div className="card card-warning">
          <FaDatabase className="card-icon" />
          <div className="card-content">
            <h3>API Endpoints</h3>
            <p className="card-value">{TOTAL_ENDPOINTS}</p>
            <p className="card-label">Endpoints available</p>
          </div>
        </div>

        <div className="card card-primary">
          <FaExclamationTriangle className="card-icon" />
          <div className="card-content">
            <h3>Device</h3>
            <p className="card-value">{healthData?.system?.device || 'cuda'}</p>
            <p className="card-label">Processing unit</p>
          </div>
        </div>
      </div>

      {/* Available Models */}
      <div className="section">
        <h2>Available AI Models</h2>
        <div className="models-grid">
          {displayedModels.map((model) => (
            <div key={model.id} className="model-card">
              <h4>{model.name}</h4>
              <span className="model-badge">{model.description}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance */}
      {metrics && metrics.models && (
        <div className="section">
          <h2>Model Performance Metrics</h2>
          <div className="metrics-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.models).map(([modelName, data]) => (
                  <tr key={modelName}>
                    <td><strong>{modelName}</strong></td>
                    <td>{(data.accuracy * 100).toFixed(2)}%</td>
                    <td>{(data.precision * 100).toFixed(2)}%</td>
                    <td>{(data.recall * 100).toFixed(2)}%</td>
                    <td>{(data.f1_score * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Disease Classes */}
      <div className="section">
        <h2>Supported Disease Classifications</h2>
        <div className="classes-grid">
          {(healthData?.data?.class_names || metrics?.class_names || DISEASE_CLASSES).map((className, index) => (
            <div key={index} className="class-badge">
              {className}
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="section">
        <h2>Quick Actions</h2>
        <div className="actions-grid">
          <button className="action-btn btn-primary" onClick={() => navigate('/predict')}>
            New Prediction
          </button>
          <button className="action-btn btn-success" onClick={() => navigate('/batch')}>
            Batch Processing
          </button>
          <button className="action-btn btn-info" onClick={() => navigate('/compare')}>
            Compare Models
          </button>
          <button className="action-btn btn-warning" onClick={() => navigate('/reports')}>
            Generate Report
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
