# Chest X-Ray AI Analysis System - Frontend

Complete frontend application for the Chest X-Ray Analysis System with **33 API Endpoints** and **6 AI Models**.

## 🚀 Features

### Complete API Integration (All 33 Endpoints)
✅ **Health & Info** (4 endpoints)
- System health check
- Detailed system status
- Model information
- Available models listing

✅ **Predictions** (6 endpoints)
- Standard prediction
- Real-time prediction
- Streaming prediction
- Batch processing
- Advanced batch processing
- Alternative batch method

✅ **Analysis & Explanation** (5 endpoints)
- AI Agent analysis with reasoning
- Prediction explanation
- Model comparison
- Multi-model comparison
- Uncertainty analysis

✅ **GradCAM/Heatmap** (6 endpoints)
- Standard GradCAM
- Enhanced GradCAM
- Heatmap generation
- GradCAM comparison
- Base64 GradCAM comparison
- Test GradCAM

✅ **Reports & Clinical** (3 endpoints)
- Standard report generation
- Clinical report with recommendations
- Patient history retrieval

✅ **File Management** (3 endpoints)
- Image upload
- File download
- Complete analysis package export

✅ **System Management** (4 endpoints)
- Model switching
- Feedback submission
- Analytics retrieval
- Performance metrics

### AI Models (All 6 Models)
1. **DenseNet-169** - Medical Standard
2. **EfficientNet-B5** - High Performance
3. **Vision Transformer (ViT-Base)** - State-of-the-Art
4. **Enhanced ViT** - Multi-Scale Attention
5. **Hybrid Model** - Multi-Model Fusion
6. **Ensemble** - Weighted Voting

### User Interface Features
- 📊 **Dashboard** - System overview and quick stats
- 🖼️ **Single Prediction** - Upload and analyze individual X-rays
- 📁 **Batch Processing** - Process multiple images at once
- 🔬 **Model Comparison** - Compare results across different models
- 🎨 **GradCAM Viewer** - Explainable AI visualizations
- 📄 **Clinical Reports** - Generate comprehensive medical reports
- 📈 **Analytics** - Performance metrics and statistics
- 📚 **History** - Patient history and past analyses
- ⚙️ **Settings** - Configuration and feedback

## 📋 Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend API running on `http://localhost:8000`

## 🛠️ Installation

1. **Extract the ZIP file**
   ```bash
   unzip chest-xray-frontend.zip
   cd chest-xray-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```
   or
   ```bash
   yarn install
   ```

3. **Configure API endpoint** (Optional)
   
   The default API URL is `http://localhost:8000`. To change it:
   - Edit `src/services/api.js` and update `API_BASE_URL`
   - Or configure it through the Settings page in the UI

## 🚀 Running the Application

### Development Mode
```bash
npm run dev
```
or
```bash
yarn dev
```

The application will start at `http://localhost:3000`

### Production Build
```bash
npm run build
```
or
```bash
yarn build
```

### Preview Production Build
```bash
npm run preview
```
or
```bash
yarn preview
```

## 📁 Project Structure

```
chest-xray-frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx           # Main dashboard
│   │   ├── PredictionPanel.jsx     # Single prediction interface
│   │   ├── BatchProcessing.jsx     # Batch processing
│   │   ├── ModelComparison.jsx     # Model comparison
│   │   ├── GradCAMViewer.jsx       # GradCAM visualizations
│   │   ├── ClinicalReports.jsx     # Report generation
│   │   ├── Analytics.jsx           # Analytics dashboard
│   │   ├── History.jsx             # Patient history
│   │   └── Settings.jsx            # Settings & configuration
│   ├── services/
│   │   └── api.js                  # API service (all 33 endpoints)
│   ├── styles/
│   │   ├── App.css                 # Main styles
│   │   └── index.css               # Global styles
│   ├── App.jsx                     # Main app component
│   └── main.jsx                    # Entry point
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## 🔧 Configuration

### Backend API Connection

1. **Using the UI:**
   - Navigate to Settings page
   - Enter your backend URL in "Backend API URL"
   - Click "Save Settings"

2. **Manually in code:**
   - Edit `src/services/api.js`
   - Change `const API_BASE_URL = 'http://localhost:8000';`

### Model Selection

You can select the default model in:
- Settings page → Default Model
- Individual prediction pages (dropdown selector)

## 📚 Usage Guide

### 1. Dashboard
- View system status and health
- Check available models
- See performance metrics
- Quick access to all features

### 2. Single Prediction
- Upload a chest X-ray image
- Select AI model
- Choose prediction type (standard/realtime/streaming/AI agent)
- View results with confidence scores
- See AI reasoning steps and medical knowledge

### 3. Batch Processing
- Upload multiple X-ray images
- Select processing model
- Choose batch type
- Download results as JSON
- View statistics and distribution

### 4. Model Comparison
- Upload one image
- Select multiple models to compare
- View side-by-side predictions
- Analyze model agreement
- Identify best performing model

### 5. GradCAM Visualization
- Upload X-ray image
- Select model and visualization type
- Generate explainable AI heatmaps
- View attention maps and important regions
- Download visualizations

### 6. Clinical Reports
- Upload X-ray and enter patient ID
- Generate comprehensive medical reports
- Export complete analysis packages
- Print or download reports

### 7. Analytics
- View system-wide statistics
- Compare model performance
- See disease distribution
- Analyze trends

### 8. History
- Search patient history by ID
- View past analyses
- Track patient timeline
- Access historical reports

### 9. Settings
- Configure default model
- Set API endpoint
- Submit feedback for model improvement
- Manage preferences

## 🎨 UI Components

### Color Scheme
- **Primary:** Blue (#3b82f6)
- **Success:** Green (#10b981)
- **Warning:** Orange (#f59e0b)
- **Danger:** Red (#ef4444)
- **Info:** Indigo (#6366f1)

### Responsive Design
- Desktop optimized (1920x1080)
- Tablet compatible (768px+)
- Mobile friendly (640px+)

## 🔌 API Endpoints Reference

All 33 endpoints are implemented in `src/services/api.js`:

```javascript
// Health & Info
healthCheck()
detailedHealthCheck()
getModels()
getModelInfo()

// Predictions
predict(formData)
predictRealtime(formData)
predictStream(formData)
predictBatch(formData)
predictBatchAdvanced(formData)
batchPredict(formData)

// Analysis
analyzeWithAgent(formData)
explainPrediction(formData)
comparePredictions(formData)
compareModels(formData)
uncertaintyAnalysis(formData)

// GradCAM
generateGradCAM(formData)
enhancedGradCAM(formData)
gradcamHeatmap(formData)
compareGradCAM(formData)
compareGradCAMBase64(formData)
testGradCAM(formData)

// Reports
generateReport(formData)
generateClinicalReport(formData)
getPatientHistory(patientId)

// Files
uploadImage(formData)
downloadFile(filename)
exportPackage(formData)

// System
switchModel(modelName)
submitFeedback(feedbackData)
getAnalytics()
getMetrics()
```

## 🐛 Troubleshooting

### Backend Connection Issues
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings on backend
- Verify firewall settings
- Check browser console for errors

### Build Issues
- Clear node_modules: `rm -rf node_modules package-lock.json`
- Reinstall: `npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`

### Display Issues
- Hard refresh: `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)
- Clear browser cache
- Try different browser

## 🤝 Support & Feedback

Use the Settings page to submit feedback about:
- Prediction accuracy
- UI/UX improvements
- Feature requests
- Bug reports

## 📊 Performance

- **Fast Load Times:** Optimized with Vite
- **Responsive UI:** React with efficient rendering
- **Real-time Updates:** WebSocket-ready architecture
- **Lazy Loading:** Components loaded on demand

## 🔒 Security

- No sensitive data stored in frontend
- API calls use secure protocols
- Patient data handled according to HIPAA guidelines
- Session-based authentication ready

## 📝 License

This project is part of the Chest X-Ray AI Analysis System.
Developed for medical imaging analysis with state-of-the-art AI models.

---

## 🎯 Quick Start Checklist

- [ ] Extract ZIP file
- [ ] Run `npm install`
- [ ] Ensure backend is running
- [ ] Run `npm run dev`
- [ ] Open `http://localhost:3000`
- [ ] Upload a test X-ray image
- [ ] Verify all features work

## 💡 Tips

1. **For best results:** Use high-quality chest X-ray images (PNG, JPG, DICOM)
2. **Model selection:** Try different models and compare results
3. **GradCAM:** Use for understanding model decisions
4. **Batch processing:** Efficient for multiple images
5. **Reports:** Export comprehensive analysis packages

---

**Enjoy analyzing chest X-rays with AI! 🏥🤖**
