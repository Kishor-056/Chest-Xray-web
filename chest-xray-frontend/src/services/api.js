import axios from 'axios';

// UPDATED: Your ngrok backend URL
const getBaseUrl = () => {
  try {
    const savedSettings = localStorage.getItem('appSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      if (settings.apiUrl) return settings.apiUrl;
    }
  } catch (e) {
    console.warn('Failed to load API URL from settings', e);
  }
  return 'https://monocyclic-shara-unrotative.ngrok-free.dev';
};

const API_BASE_URL = getBaseUrl();
const STREAM_ENDPOINT = `${API_BASE_URL}/predict/stream`;
const WS_ENDPOINT = `${API_BASE_URL.replace('http', 'ws')}/ws/realtime`;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
  },
});

// Add request interceptor for retry logic
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching issues
    config.params = { ...config.params, _t: Date.now() };
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;

    // Retry logic for network errors
    if (!config || !config.retry) {
      config.retry = 0;
    }

    // Retry up to 2 times for network errors
    if (config.retry < 2 && (!error.response || error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK')) {
      config.retry += 1;
      console.warn(`Retrying request (${config.retry}/2):`, config.url);

      // Wait before retrying (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, 1000 * config.retry));

      return api(config);
    }

    // Enhanced error message
    if (!error.response) {
      error.message = 'Cannot connect to backend server. Please check if the backend is running.';
    } else if (error.response.status === 404) {
      error.message = `Endpoint not found: ${config.url}`;
    } else if (error.response.status >= 500) {
      error.message = 'Backend server error. Please try again later.';
    }

    return Promise.reject(error);
  }
);

export const TOTAL_ENDPOINTS = 33;

// ============================================================================
// HEALTH & INFO ENDPOINTS (5)
// ============================================================================

export const healthCheck = () => api.get('/');
export const detailedHealthCheck = () => api.get('/health/detailed');
export const getModels = () => api.get('/models');
export const getModelInfo = () => api.get('/model/info');
export const getSpecificModelInfo = (modelName) => api.get(`/info/${modelName}`);

// X-ray Validation
export const validateXray = (formData) => api.post('/validate/xray', formData, {
  headers: {
    'Content-Type': 'multipart/form-data',
    'ngrok-skip-browser-warning': 'true',
  },
});

// ============================================================================
// PREDICTION ENDPOINTS (6)
// ============================================================================

const multipartConfig = {
  headers: {
    'Content-Type': 'multipart/form-data',
    'ngrok-skip-browser-warning': 'true',
  },
};

export const predict = (formData) => api.post('/predict', formData, multipartConfig);

export const predictRealtime = (formData) => api.post('/predict/realtime', formData, multipartConfig);

export const predictStream = async (formData, onMessage) => {
  const response = await fetch(STREAM_ENDPOINT, {
    method: 'POST',
    body: formData,
    headers: {
      'ngrok-skip-browser-warning': 'true',
    },
  });

  if (!response.body) {
    throw new Error('Streaming is not supported by the current browser.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let latest = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      const message = JSON.parse(line);
      latest = message;
      onMessage?.(message);
    }
  }

  if (buffer.trim()) {
    const message = JSON.parse(buffer.trim());
    latest = message;
    onMessage?.(message);
  }

  return latest;
};

export const predictBatch = (formData) => api.post('/predict/batch', formData, multipartConfig);

export const predictBatchAdvanced = (formData) => api.post('/predict/batch/advanced', formData, multipartConfig);

export const batchPredict = (formData) => api.post('/batch_predict', formData, multipartConfig);

// ============================================================================
// ANALYSIS & EXPLANATION ENDPOINTS (5)
// ============================================================================

export const analyzeWithAgent = (formData) => api.post('/analyze', formData, multipartConfig);

export const explainPrediction = (formData) => api.post('/explain', formData, multipartConfig);

export const comparePredictions = (formData) => api.post('/compare', formData, multipartConfig);

export const compareModels = (formData) => api.post('/compare_models', formData, multipartConfig);

export const uncertaintyAnalysis = (formData) => api.post('/uncertainty', formData, multipartConfig);

// ============================================================================
// GRADCAM/HEATMAP ENDPOINTS (6)
// ============================================================================

export const generateGradCAM = (formData) => api.post('/gradcam', formData, multipartConfig);

export const testGradCAM = (formData) => api.post('/gradcam/test', formData, multipartConfig);

export const enhancedGradCAM = (formData) => api.post('/gradcam/enhanced', formData, multipartConfig);

export const gradcamHeatmap = (formData) => api.post('/gradcam/heatmap', formData, multipartConfig);

export const compareGradCAM = (formData) => api.post('/gradcam/compare', formData, multipartConfig);

export const compareGradCAMBase64 = (formData) => api.post('/gradcam/compare/base64', formData, multipartConfig);

// ============================================================================
// REPORTS & CLINICAL ENDPOINTS (3)
// ============================================================================

export const generateReport = (formData) => api.post('/report', formData, multipartConfig);

export const generateClinicalReport = (formData) => api.post('/clinical_report', formData, multipartConfig);

export const getPatientHistory = (patientId) => api.get(`/history/${patientId}`);

// ============================================================================
// FILE MANAGEMENT ENDPOINTS (3)
// ============================================================================

export const uploadImage = (formData) => api.post('/upload', formData, multipartConfig);

export const downloadFile = (filename) => api.get(`/download/${filename}`, {
  responseType: 'blob',
});

export const exportPackage = (formData) => api.post(
  '/export/package',
  formData,
  { ...multipartConfig, responseType: 'blob' }
);

// ============================================================================
// SYSTEM MANAGEMENT ENDPOINTS (4)
// ============================================================================

export const switchModel = (modelName) => api.post(
  '/model/switch',
  new URLSearchParams({ model_name: modelName }),
  { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
);

export const submitFeedback = ({ predictionId, correctLabel, confidence, notes }) => {
  const payload = new URLSearchParams();

  if (predictionId) payload.append('prediction_id', predictionId);
  if (correctLabel) payload.append('correct_label', correctLabel);
  if (typeof confidence === 'number') payload.append('confidence', confidence.toString());
  if (notes) payload.append('notes', notes);

  return api.post('/feedback', payload, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
};

export const getAnalytics = () => api.get('/analytics');

export const getMetrics = () => api.get('/metrics');

export const connectRealtimeWebSocket = () => new WebSocket(WS_ENDPOINT);

// ============================================================================
// MODELS CONFIGURATION
// ============================================================================

export const CORE_MODELS = [
  { id: 'DenseNet169', name: 'DenseNet-169', description: 'High-capacity CNN baseline' },
  { id: 'EfficientNet-B5', name: 'EfficientNet-B5', description: 'Lightweight high-accuracy CNN' },
  { id: 'ViT-Base', name: 'ViT-Base', description: 'Vision Transformer base model' },
  { id: 'ViT-Base-Enhanced', name: 'ViT-Base Enhanced', description: 'Transformer with advanced XAI' },
  { id: 'Enhanced-Hybrid', name: 'Enhanced Hybrid', description: 'DenseNet + EfficientNet + VIT BASE' },
];

export const ENSEMBLE_OPTION = {
  id: 'ensemble',
  name: 'Ensemble',
  description: 'Weighted fusion across all five models',
};

export const MODEL_OPTIONS = [...CORE_MODELS, ENSEMBLE_OPTION];

export const DISEASE_CLASSES = [
  'COVID-19',
  'Normal',
  'Pneumonia',
  'Tuberculosis',
];

export default api;