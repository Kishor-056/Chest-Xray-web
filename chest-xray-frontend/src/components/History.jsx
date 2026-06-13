import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaSearch, FaEye, FaDownload } from 'react-icons/fa';
import { getPatientHistory } from '../services/api';

function History() {
  const [patientId, setPatientId] = useState('');
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearchHistory = async () => {
    if (!patientId.trim()) {
      toast.error('Please enter a patient ID');
      return;
    }

    setLoading(true);
    setHistory(null);

    try {
      const response = await getPatientHistory(patientId);
      setHistory(response.data);
      toast.success('History loaded successfully');
    } catch (error) {
      console.error('Failed to load history:', error);
      if (error.response?.status === 404) {
        toast.warning('No history found for this patient ID');
      } else {
        toast.error('Failed to load patient history');
      }
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="history">
      <div className="panel-header">
        <h1>Patient History</h1>
        <p>View past analysis results and reports</p>
      </div>

      <div className="history-content">
        {/* Search Section */}
        <div className="search-section">
          <div className="search-bar">
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="Enter Patient ID..."
              onKeyPress={(e) => e.key === 'Enter' && handleSearchHistory()}
            />
            <button 
              className="btn-search"
              onClick={handleSearchHistory}
              disabled={loading}
            >
              <FaSearch /> Search
            </button>
          </div>
        </div>

        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading history...</p>
          </div>
        )}

        {/* History Results */}
        {history && (
          <div className="history-results">
            <div className="history-header">
              <h2>History for Patient: {patientId}</h2>
              <p className="record-count">
                {history.records?.length || 0} record(s) found
              </p>
            </div>

            {history.records && history.records.length > 0 ? (
              <div className="records-timeline">
                {history.records.map((record, index) => (
                  <div key={index} className="record-card">
                    <div className="record-header">
                      <div className="record-date">
                        <span className="date-icon">📅</span>
                        <span>{formatDate(record.timestamp || record.date)}</span>
                      </div>
                      <div className="record-actions">
                        <button className="btn-icon" title="View Details">
                          <FaEye />
                        </button>
                        <button className="btn-icon" title="Download">
                          <FaDownload />
                        </button>
                      </div>
                    </div>

                    <div className="record-body">
                      <div className="record-diagnosis">
                        <h4>Diagnosis</h4>
                        <span className="diagnosis-badge">{record.prediction || record.diagnosis}</span>
                        <span className="confidence-tag">
                          {((record.confidence || 0) * 100).toFixed(1)}% confidence
                        </span>
                      </div>

                      {record.model && (
                        <div className="record-field">
                          <strong>Model Used:</strong> {record.model}
                        </div>
                      )}

                      {record.findings && (
                        <div className="record-field">
                          <strong>Findings:</strong> {record.findings}
                        </div>
                      )}

                      {record.recommendations && (
                        <div className="record-field">
                          <strong>Recommendations:</strong>
                          <ul className="recommendations-list">
                            {record.recommendations.map((rec, i) => (
                              <li key={i}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-records">
                <p>No history records found for this patient.</p>
              </div>
            )}

            {/* Summary Statistics */}
            {history.summary && (
              <div className="history-summary">
                <h3>Summary Statistics</h3>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Total Scans</span>
                    <span className="summary-value">{history.summary.total_scans}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Most Common Diagnosis</span>
                    <span className="summary-value">{history.summary.most_common}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Average Confidence</span>
                    <span className="summary-value">
                      {(history.summary.avg_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Local Storage History (as fallback) */}
        <div className="local-history-section">
          <h3>Recent Local Activity</h3>
          <p className="info-text">
            Recent predictions from this session (stored locally)
          </p>
          <div className="local-history-placeholder">
            <p>No local history available</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default History;
