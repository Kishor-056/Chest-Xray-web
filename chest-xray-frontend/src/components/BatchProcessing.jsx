import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaUpload, FaSpinner, FaDownload, FaTrash } from 'react-icons/fa';
import { predictBatch, predictBatchAdvanced, batchPredict, MODEL_OPTIONS } from '../services/api';
import { validateImageLocally } from '../utils/validation';

function BatchProcessing() {
  const [files, setFiles] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [batchType, setBatchType] = useState('standard');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFilesChange = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    toast.info(`Validating ${selectedFiles.length} image(s) with AI...`);

    const validFiles = [];
    let rejectedCount = 0;

    for (const file of selectedFiles) {
      const validation = await validateImageLocally(file);
      if (validation.is_valid) {
        validFiles.push(file);
      } else {
        rejectedCount++;
        toast.error(`❌ ${file.name}: ${validation.message}`, { autoClose: 4000 });
      }
    }

    if (rejectedCount > 0) {
      toast.warning(`${rejectedCount} non-X-ray file(s) were rejected. ${validFiles.length} valid image(s) accepted.`);
    }

    if (validFiles.length > 0) {
      setFiles(validFiles);
      toast.success(`✅ ${validFiles.length} valid chest X-ray(s) ready for batch processing.`);
    } else {
      toast.error('No valid chest X-ray images found. Please upload medical X-ray images.');
    }
  };

  const handleBatchProcess = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one image');
      return;
    }

    setLoading(true);
    setResults(null);
    setProgress(0);

    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append('files', file);
      });
      formData.append('model_name', selectedModel);

      let response;
      switch (batchType) {
        case 'advanced':
          response = await predictBatchAdvanced(formData);
          break;
        case 'alternative':
          response = await batchPredict(formData);
          break;
        default:
          response = await predictBatch(formData);
      }

      setResults(response.data);
      setProgress(100);
      toast.success(`Batch processing completed! ${files.length} images analyzed`);
    } catch (error) {
      console.error('Batch processing failed:', error);
      toast.error(error.response?.data?.detail || 'Batch processing failed');
    } finally {
      setLoading(false);
    }
  };

  const clearFiles = () => {
    setFiles([]);
    setResults(null);
    setProgress(0);
  };

  const downloadResults = () => {
    if (!results) return;

    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch_results_${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Results downloaded');
  };

  return (
    <div className="batch-processing">
      <div className="panel-header">
        <h1>Batch Processing</h1>
        <p>Process multiple chest X-rays simultaneously</p>
      </div>

      <div className="batch-content">
        {/* Upload Section */}
        <div className="batch-upload-section">
          <div className="upload-controls">
            <label className="file-input-label">
              <FaUpload /> Select Multiple Images
              <input
                type="file"
                multiple
                accept="image/*,.dcm"
                onChange={handleFilesChange}
                style={{ display: 'none' }}
              />
            </label>
            {files.length > 0 && (
              <button className="btn-clear" onClick={clearFiles}>
                <FaTrash /> Clear All
              </button>
            )}
          </div>

          {/* Files Preview */}
          {files.length > 0 && (
            <div className="files-list">
              <h3>Selected Files ({files.length})</h3>
              <div className="files-grid">
                {files.map((file, index) => (
                  <div key={index} className="file-item">
                    <div className="file-icon">📄</div>
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Configuration */}
          <div className="batch-config">
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
              <label>Batch Type</label>
              <select value={batchType} onChange={(e) => setBatchType(e.target.value)}>
                <option value="standard">Standard Batch</option>
                <option value="advanced">Advanced Batch</option>
                <option value="alternative">Alternative Method</option>
              </select>
            </div>

            <button
              className="btn-process"
              onClick={handleBatchProcess}
              disabled={files.length === 0 || loading}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinner" /> Processing {files.length} images...
                </>
              ) : (
                `Process ${files.length} Images`
              )}
            </button>
          </div>

          {/* Progress Bar */}
          {loading && (
            <div className="progress-section">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="progress-text">Processing... {progress}%</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="batch-results">
            <div className="results-header">
              <h2>Batch Results</h2>
              <button className="btn-download" onClick={downloadResults}>
                <FaDownload /> Download Results
              </button>
            </div>

            {/* Statistics */}
            {results.statistics && (
              <div className="statistics-grid">
                <div className="stat-card">
                  <h4>Total Processed</h4>
                  <p className="stat-value">{results.statistics.total || files.length}</p>
                </div>
                <div className="stat-card">
                  <h4>Processing Time</h4>
                  <p className="stat-value">{results.processing_time?.toFixed(2)}s</p>
                </div>
                <div className="stat-card">
                  <h4>Success Rate</h4>
                  <p className="stat-value">
                    {results.statistics.success_rate
                      ? `${(results.statistics.success_rate * 100).toFixed(1)}%`
                      : '100%'}
                  </p>
                </div>
                <div className="stat-card">
                  <h4>Average Confidence</h4>
                  <p className="stat-value">
                    {results.statistics.avg_confidence
                      ? `${(results.statistics.avg_confidence * 100).toFixed(1)}%`
                      : 'N/A'}
                  </p>
                </div>
              </div>
            )}

            {/* Individual Results */}
            {results.results && (
              <div className="results-table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Image</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.results.map((result, index) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td className="image-name">{result.filename || files[index]?.name}</td>
                        <td>
                          <span className="prediction-badge">{result.prediction}</span>
                        </td>
                        <td>
                          <div className="confidence-cell">
                            <div className="mini-bar">
                              <div
                                className="mini-fill"
                                style={{ width: `${(result.confidence * 100)}%` }}
                              ></div>
                            </div>
                            <span>{(result.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td>
                          <span className="status-badge status-success">✓ Complete</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Disease Distribution */}
            {results.statistics && results.statistics.disease_distribution && (
              <div className="distribution-section">
                <h3>Disease Distribution</h3>
                <div className="distribution-chart">
                  {Object.entries(results.statistics.disease_distribution).map(([disease, count]) => (
                    <div key={disease} className="distribution-bar">
                      <span className="disease-label">{disease}</span>
                      <div className="bar-container">
                        <div
                          className="bar-fill"
                          style={{ width: `${(count / files.length) * 100}%` }}
                        ></div>
                      </div>
                      <span className="count-label">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default BatchProcessing;
