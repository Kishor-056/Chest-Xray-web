import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { 
  FaHome, FaImage, FaChartBar, FaFlask, FaFileAlt, 
  FaCog, FaHistory, FaLayerGroup, FaHeartbeat 
} from 'react-icons/fa';

// Import components
import Dashboard from './components/Dashboard';
import PredictionPanel from './components/PredictionPanel';
import BatchProcessing from './components/BatchProcessing';
import ModelComparison from './components/ModelComparison';
import GradCAMViewer from './components/GradCAMViewer';
import ClinicalReports from './components/ClinicalReports';
import Analytics from './components/Analytics';
import Settings from './components/Settings';
import History from './components/History';

import { healthCheck, MODEL_OPTIONS } from './services/api';
import './styles/App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await healthCheck();
      setSystemStatus(response.data);
      setLoading(false);
    } catch (error) {
      console.error('System health check failed:', error);
      toast.error('Backend server is not responding. Please check connection.');
      setLoading(false);
    }
  };

  return (
    <Router>
      <div className="app">
        <ToastContainer position="top-right" autoClose={3000} />
        
        {/* Sidebar Navigation */}
        <nav className="sidebar">
          <div className="sidebar-header">
            <FaHeartbeat className="logo-icon" />
            <h2>Chest X-Ray AI</h2>
            <div className="status-indicator">
              <span className={`status-dot ${systemStatus?.status === 'healthy' ? 'online' : 'offline'}`}></span>
              <span className="status-text">
                {systemStatus?.status === 'healthy' ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>

          <ul className="nav-menu">
            <li>
              <Link to="/" className="nav-link">
                <FaHome /> Dashboard
              </Link>
            </li>
            <li>
              <Link to="/predict" className="nav-link">
                <FaImage /> Single Prediction
              </Link>
            </li>
            <li>
              <Link to="/batch" className="nav-link">
                <FaLayerGroup /> Batch Processing
              </Link>
            </li>
            <li>
              <Link to="/compare" className="nav-link">
                <FaFlask /> Model Comparison
              </Link>
            </li>
            <li>
              <Link to="/gradcam" className="nav-link">
                <FaChartBar /> GradCAM Viewer
              </Link>
            </li>
            <li>
              <Link to="/reports" className="nav-link">
                <FaFileAlt /> Clinical Reports
              </Link>
            </li>
            <li>
              <Link to="/analytics" className="nav-link">
                <FaChartBar /> Analytics
              </Link>
            </li>
            <li>
              <Link to="/history" className="nav-link">
                <FaHistory /> History
              </Link>
            </li>
            <li>
              <Link to="/settings" className="nav-link">
                <FaCog /> Settings
              </Link>
            </li>
          </ul>

          <div className="sidebar-footer">
            <div className="system-info">
              <p><strong>Models:</strong> {systemStatus?.models_loaded || MODEL_OPTIONS.length}</p>
              <p><strong>Classes:</strong> {systemStatus?.classes || 4}</p>
              <p><strong>Device:</strong> {systemStatus?.device || 'cuda'}</p>
            </div>
          </div>
        </nav>

        {/* Main Content Area */}
        <main className="main-content">
          {loading ? (
            <div className="loading-screen">
              <div className="spinner"></div>
              <p>Connecting to AI Backend...</p>
            </div>
          ) : (
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<PredictionPanel />} />
              <Route path="/batch" element={<BatchProcessing />} />
              <Route path="/compare" element={<ModelComparison />} />
              <Route path="/gradcam" element={<GradCAMViewer />} />
              <Route path="/reports" element={<ClinicalReports />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          )}
        </main>
      </div>
    </Router>
  );
}

export default App;
