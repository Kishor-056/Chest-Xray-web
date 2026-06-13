import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { FaUpload, FaSpinner, FaDownload, FaFilePdf } from 'react-icons/fa';
import { 
  generateReport, 
  generateClinicalReport,
  exportPackage,
  MODEL_OPTIONS 
} from '../services/api';
import { saveAs } from 'file-saver';

function ClinicalReports() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [patientId, setPatientId] = useState('');
  const [reportType, setReportType] = useState('clinical');
  const [includeGradCAM, setIncludeGradCAM] = useState(true);
  const [report, setReport] = useState(null);
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

  const handleGenerateReport = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);
    setReport(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', selectedModel);
      if (patientId) formData.append('patient_id', patientId);

      let response;
      if (reportType === 'clinical') {
        response = await generateClinicalReport(formData);
      } else {
        response = await generateReport(formData);
      }

      setReport(response.data);
      toast.success('Report generated successfully!');
    } catch (error) {
      console.error('Report generation failed:', error);
      toast.error(error.response?.data?.detail || 'Report generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExportPackage = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', selectedModel);
      formData.append('include_gradcam', includeGradCAM);

      const response = await exportPackage(formData);

      const blob = response.data instanceof Blob
        ? response.data
        : new Blob([response.data], { type: 'application/zip' });
      saveAs(blob, `analysis_package_${Date.now()}.zip`);
      
      toast.success('Analysis package downloaded!');
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export package');
    } finally {
      setLoading(false);
    }
  };

  const downloadReportJSON = () => {
    if (!report) return;
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    saveAs(dataBlob, `report_${patientId || 'anonymous'}_${Date.now()}.json`);
    toast.success('Report downloaded');
  };

  const printReport = () => {
    window.print();
  };

  return (
    <div className="clinical-reports">
      <div className="panel-header">
        <h1>Clinical Reports</h1>
        <p>Generate comprehensive medical reports with AI analysis</p>
      </div>

      <div className="reports-content">
        {/* Input Section */}
        <div className="report-input-section">
          <div className="upload-area" onClick={() => document.getElementById('reportFileInput').click()}>
            {preview ? (
              <img src={preview} alt="Preview" className="image-preview" />
            ) : (
              <div className="upload-placeholder">
                <FaUpload className="upload-icon" />
                <p>Click to upload chest X-ray</p>
              </div>
            )}
            <input
              id="reportFileInput"
              type="file"
              accept="image/*,.dcm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>

          <div className="report-config">
            <div className="form-group">
              <label>Patient ID (Optional)</label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g., P12345"
              />
            </div>

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
              <label>Report Type</label>
              <select value={reportType} onChange={(e) => setReportType(e.target.value)}>
                <option value="clinical">Clinical Report</option>
                <option value="standard">Standard Report</option>
              </select>
            </div>

            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={includeGradCAM}
                  onChange={(e) => setIncludeGradCAM(e.target.checked)}
                />
                Include GradCAM Visualization
              </label>
            </div>

            <div className="button-group">
              <button 
                className="btn-generate-report"
                onClick={handleGenerateReport}
                disabled={!selectedFile || loading}
              >
                {loading ? (
                  <>
                    <FaSpinner className="spinner" /> Generating...
                  </>
                ) : (
                  'Generate Report'
                )}
              </button>

              <button 
                className="btn-export"
                onClick={handleExportPackage}
                disabled={!selectedFile || loading}
              >
                <FaFilePdf /> Export Package
              </button>
            </div>
          </div>
        </div>

        {/* Report Display */}
        {report && (
          <div className="report-display">
            <div className="report-actions">
              <button className="btn-action" onClick={downloadReportJSON}>
                <FaDownload /> Download JSON
              </button>
              <button className="btn-action" onClick={printReport}>
                🖨️ Print Report
              </button>
            </div>

            <div className="report-content printable">
              {/* Report Header */}
              <div className="report-header">
                <h2>Medical Imaging Report</h2>
                <div className="report-meta">
                  {report.patient_id && <p><strong>Patient ID:</strong> {report.patient_id}</p>}
                  <p><strong>Report Date:</strong> {new Date(report.timestamp).toLocaleString()}</p>
                  <p><strong>Analysis Type:</strong> Chest X-Ray AI Analysis</p>
                </div>
              </div>

              {/* Diagnosis Section */}
              <div className="report-section">
                <h3>Primary Diagnosis</h3>
                <div className="diagnosis-box">
                  <p className="diagnosis-label">{report.prediction}</p>
                  <p className="confidence-text">
                    Confidence Level: {(report.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {/* Risk Assessment */}
              {report.risk_assessment && (
                <div className="report-section">
                  <h3>Risk Assessment</h3>
                  <div className="risk-box">
                    <p><strong>Risk Level:</strong> {report.risk_assessment.risk_level}</p>
                    {report.risk_assessment.severity && (
                      <p><strong>Severity:</strong> {report.risk_assessment.severity}</p>
                    )}
                    {report.risk_assessment.urgency && (
                      <p><strong>Urgency:</strong> {report.risk_assessment.urgency}</p>
                    )}
                  </div>
                </div>
              )}

              {/* Clinical Notes */}
              {report.clinical_notes && report.clinical_notes.length > 0 && (
                <div className="report-section">
                  <h3>Clinical Notes</h3>
                  <ul className="notes-list">
                    {report.clinical_notes.map((note, index) => (
                      <li key={index}>{note}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recommended Actions */}
              {report.recommended_actions && report.recommended_actions.length > 0 && (
                <div className="report-section">
                  <h3>Recommended Actions</h3>
                  <ul className="actions-list">
                    {report.recommended_actions.map((action, index) => (
                      <li key={index}>{action}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Findings */}
              {report.findings && (
                <div className="report-section">
                  <h3>Key Findings</h3>
                  <p>{report.findings}</p>
                </div>
              )}

              {/* Probabilities */}
              {report.probabilities && (
                <div className="report-section">
                  <h3>Differential Diagnosis Probabilities</h3>
                  <div className="probabilities-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Condition</th>
                          <th>Probability</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(report.probabilities).map(([condition, prob]) => (
                          <tr key={condition}>
                            <td>{condition}</td>
                            <td>{(prob * 100).toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Model Information */}
              <div className="report-section">
                <h3>Analysis Details</h3>
                <div className="analysis-details">
                  <p><strong>AI Model Used:</strong> {report.model_used || selectedModel}</p>
                  {report.model_weights && (
                    <div className="model-weights">
                      <p><strong>Model Fusion Weights:</strong></p>
                      {Object.entries(report.model_weights).map(([model, weight]) => (
                        <p key={model}>• {model}: {(weight * 100).toFixed(1)}%</p>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Visualization Link */}
              {report.visualization_url && (
                <div className="report-section">
                  <h3>Visualization</h3>
                  <p>GradCAM visualization available at: {report.visualization_url}</p>
                </div>
              )}

              {/* Disclaimer */}
              <div className="report-section disclaimer">
                <h4>⚠️ Medical Disclaimer</h4>
                <p>
                  This report is generated by an AI system and should be used as a decision support tool only. 
                  All diagnoses must be confirmed by qualified medical professionals. This analysis does not 
                  replace professional medical judgment and should not be used as the sole basis for medical decisions.
                </p>
              </div>

              {/* Footer */}
              <div className="report-footer">
                <p>Report generated by Chest X-Ray AI Analysis System</p>
                <p>Version 1.0 | {new Date().toLocaleDateString()}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ClinicalReports;
