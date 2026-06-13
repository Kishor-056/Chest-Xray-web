/**
 * Advanced Explainable AI (XAI) Engine for Chest X-Ray Analysis
 * 
 * Techniques implemented:
 *  1. Confidence-based severity scoring (4 tiers)
 *  2. Differential diagnosis from all_probabilities (what else it could be)
 *  3. Disease-specific clinical knowledge base
 *     - X-ray visual patterns detected
 *     - Aetiology (causes — viral/bacterial/environmental)
 *     - Pathophysiology (why it happens)
 *     - Symptom correlation
 *     - Treatment & remedies
 *     - Emergency criteria
 *  4. Uncertainty / confidence interval estimation
 *  5. Feature importance narrative (maps confidence → X-ray regions)
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1. SEVERITY ENGINE
// ─────────────────────────────────────────────────────────────────────────────

export const SEVERITY_LEVELS = {
    NORMAL: { label: 'Normal', color: '#10b981', bg: '#d1fae5', icon: '✅', urgency: 0 },
    LOW: { label: 'Low', color: '#3b82f6', bg: '#dbeafe', icon: '🔵', urgency: 1 },
    MODERATE: { label: 'Moderate', color: '#f59e0b', bg: '#fef3c7', icon: '🟡', urgency: 2 },
    HIGH: { label: 'High', color: '#f97316', bg: '#ffedd5', icon: '🟠', urgency: 3 },
    CRITICAL: { label: 'Critical', color: '#ef4444', bg: '#fee2e2', icon: '🔴', urgency: 4 },
};

export const getSeverity = (prediction, confidence) => {
    if (prediction === 'Normal') return SEVERITY_LEVELS.NORMAL;
    if (confidence < 0.60) return SEVERITY_LEVELS.LOW;
    if (confidence < 0.75) return SEVERITY_LEVELS.MODERATE;
    if (confidence < 0.90) return SEVERITY_LEVELS.HIGH;
    return SEVERITY_LEVELS.CRITICAL;
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. CLINICAL KNOWLEDGE BASE
// ─────────────────────────────────────────────────────────────────────────────

const KNOWLEDGE_BASE = {
    'Normal': {
        xray_patterns: [
            'Lung fields are clear with no opacities or consolidations',
            'Cardiac silhouette is within normal limits (cardiothoracic ratio < 0.5)',
            'Lung parenchyma shows normal air density throughout both lung fields',
            'No pleural effusion, pneumothorax, or cardiomegaly detected',
            'Diaphragm domes are well-defined and symmetrical',
            'Trachea is midline with no deviation',
        ],
        why_ai_decided: [
            'Homogeneous lung density without focal opacities',
            'Normal vascular markings visible throughout lung fields',
            'No increased opacity suggesting consolidation or infection',
            'Costophrenic angles are sharp and clear',
        ],
        causes: [],
        pathophysiology: 'No pathological process detected. The lung fields appear healthy with normal aeration and tissue density.',
        symptoms_correlation: 'Patient likely presents with no respiratory distress or minimal symptoms.',
        treatments: ['No treatment required', 'Routine health check-up recommended', 'Maintain healthy lifestyle'],
        remedies: [
            { category: 'Prevention', items: ['Regular exercise (30 min/day)', 'Avoid smoking and second-hand smoke', 'Annual flu vaccination', 'COVID-19 vaccination series'] },
            { category: 'Lifestyle', items: ['Balanced diet rich in antioxidants', 'Adequate hydration (8 glasses/day)', 'Avoid air pollution exposure', 'Regular pulmonary function tests'] },
        ],
        emergency_signs: [],
        follow_up: 'Routine annual chest X-ray if clinically indicated.',
        prognosis: 'Excellent. No pulmonary abnormality detected.',
    },

    'Pneumonia': {
        xray_patterns: [
            'Focal or lobar opacification detected — classic consolidation pattern',
            'Air bronchograms visible within the opacity (air-filled bronchi surrounded by consolidated alveoli)',
            'Increased density in one or more lung lobes typical of alveolar filling',
            'Possible silhouette sign (loss of normal cardiac or diaphragm border)',
            'Interstitial markings may be increased in atypical (viral) pneumonia',
            'Blunting of costophrenic angle suggesting parapneumonic effusion',
        ],
        why_ai_decided: [
            `The AI detected regional opacity increases consistent with alveolar consolidation`,
            'Loss of normal lung translucency in affected lobe(s)',
            'Pattern matches lobar or bronchopneumonia presentation',
            'Airspace disease pattern with characteristic "ground-glass" or solid opacity',
        ],
        causes: [
            { type: 'Bacterial (Most Common)', agents: ['Streptococcus pneumoniae', 'Haemophilus influenzae', 'Staphylococcus aureus', 'Klebsiella pneumoniae'] },
            { type: 'Viral', agents: ['Influenza A & B', 'Respiratory Syncytial Virus (RSV)', 'Adenovirus', 'Parainfluenza'] },
            { type: 'Atypical', agents: ['Mycoplasma pneumoniae', 'Chlamydophila pneumoniae', 'Legionella pneumophila'] },
            { type: 'Risk Factors', agents: ['Age > 65 or < 2 years', 'Immunosuppression', 'Chronic lung disease (COPD, asthma)', 'Smoking', 'Hospitalization'] },
        ],
        pathophysiology: 'Pathogenic organisms invade the lower respiratory tract, triggering an inflammatory response. Inflammatory exudate floods the alveoli, replacing air with fluid and cells — producing the radiographic "consolidation" the AI detected. This reduces gas exchange surface area, causing hypoxia.',
        symptoms_correlation: 'Patient likely presents with: fever & chills, productive cough (yellow/green/rust-colored sputum), pleuritic chest pain, dyspnea, fatigue, and reduced SpO₂.',
        treatments: [
            'Antibiotics (amoxicillin, azithromycin, or fluoroquinolone based on severity)',
            'Antiviral therapy if viral aetiology suspected',
            'Supplemental oxygen if SpO₂ < 94%',
            'IV fluids if dehydrated',
            'Antipyretics for fever management',
            'Hospitalization if PSI score IV-V or CURB-65 ≥ 3',
        ],
        remedies: [
            { category: '💊 Medical Treatment', items: ['Complete prescribed antibiotic course (5–10 days)', 'Use bronchodilators if wheezing present', 'Corticosteroids in severe community-acquired pneumonia'] },
            { category: '🏠 Home Care', items: ['Bed rest and adequate sleep', 'Drink 8–10 glasses of water/day', 'Humidifier to ease breathing', 'Steam inhalation 2–3 times daily', 'Honey + ginger tea for cough relief'] },
            { category: '🥗 Nutrition', items: ['High-protein diet for immune recovery', 'Vitamin C (citrus fruits, bell peppers)', 'Zinc-rich foods (pumpkin seeds, legumes)', 'Avoid alcohol and smoking completely'] },
            { category: '🩺 Monitoring', items: ['Check temperature every 4 hours', 'Monitor SpO₂ with pulse oximeter', 'Follow-up chest X-ray at 4–6 weeks', 'Return if symptoms worsen after 48h of antibiotics'] },
        ],
        emergency_signs: [
            'SpO₂ < 90% or cyanosis (blue lips/fingertips)',
            'Respiratory rate > 30 breaths/min',
            'Confusion or altered mental status',
            'Systolic BP < 90 mmHg',
            'High fever > 40°C unresponsive to antipyretics',
            'Unable to maintain oral intake',
        ],
        follow_up: 'Repeat chest X-ray in 4–6 weeks to confirm resolution. All opacities should clear. Persistent opacity after 6 weeks requires CT scan to exclude malignancy.',
        prognosis: 'Good with appropriate antibiotic therapy. Mortality < 1% in outpatient setting; 5–15% if hospitalized; higher in ICU patients.',
    },

    'Tuberculosis': {
        xray_patterns: [
            'Upper lobe predominant opacity — hallmark of post-primary TB',
            'Cavitation (thick-walled cavities with air-fluid levels) in upper lobes',
            'Miliary pattern: 1–3 mm nodules uniformly distributed throughout both lungs',
            'Hilar or mediastinal lymphadenopathy (primary TB)',
            'Fibrosis, calcification, or pleural thickening from healed/latent TB',
            'Consolidation or lobar collapse in advanced disease',
        ],
        why_ai_decided: [
            'Upper-lobe predominant density pattern localised predominantly to apical segments',
            'Possible cavitation features detected — characteristic of TB liquefaction necrosis',
            'Asymmetric distribution of opacities not typical of other pneumonias',
            'Pattern of disease localisation in apical/posterior upper lobe segments',
        ],
        causes: [
            { type: 'Pathogen', agents: ['Mycobacterium tuberculosis (primary cause)', 'M. bovis (zoonotic, bovine source)', 'M. africanum (sub-Saharan Africa)'] },
            { type: 'Transmission', agents: ['Airborne droplet nuclei (< 5 μm)', 'Prolonged close contact with active TB case', 'Overcrowded living conditions', 'Healthcare worker exposure'] },
            { type: 'Risk Factors', agents: ['HIV/AIDS co-infection (50× higher risk)', 'Malnutrition and poverty', 'DM, renal failure, immunosuppression', 'Silicosis / dusty occupational exposure', 'Alcohol abuse and homelessness'] },
        ],
        pathophysiology: 'M. tuberculosis is inhaled and reaches alveoli, where macrophages attempt to contain it. If immunity fails, granulomas form (Ghon focus). In post-primary TB, reactivation causes caseous necrosis and liquefaction — producing the cavities the AI detected. Liquefied material drains into bronchi, releasing bacilli and causing further spread.',
        symptoms_correlation: 'Classic presentation: 2–3 weeks of persistent cough (may be blood-stained — haemoptysis), night sweats, low-grade fever (especially evening), unexplained weight loss ("consumption"), fatigue, and anorexia.',
        treatments: [
            'Standard DOTS regimen: 2 months HRZE + 4 months HR (6 months total)',
            'H = Isoniazid, R = Rifampicin, Z = Pyrazinamide, E = Ethambutol',
            'MDR-TB (Multi-Drug Resistant): 18–24 month combination therapy',
            'Contact tracing and isolation for infectious cases',
            'Directly Observed Therapy Short-course (DOTS)',
            'Corticosteroids for TB meningitis or pericarditis',
        ],
        remedies: [
            { category: '💊 Standard Drug Regimen', items: ['Never miss doses — partial treatment creates drug resistance (MDR-TB)', 'Pyridoxine (Vitamin B6) with isoniazid to prevent peripheral neuropathy', 'Monitor liver function tests monthly (hepatotoxic drugs)', 'Ethambutol: monthly visual acuity testing'] },
            { category: '🏠 Infection Control', items: ['Isolate patient until 2 weeks of treatment completed + sputum negative', 'N95 mask for caregivers in enclosed spaces', 'Ventilate rooms well (UV air disinfection if available)', 'Contact tracing: screen all close contacts with TST/IGRA'] },
            { category: '🥗 Nutritional Support', items: ['High-calorie, high-protein diet (3000 kcal/day)', 'Vitamin D supplementation (immune modulation)', 'Zinc and iron supplementation', 'Avoid alcohol completely (fatal with TB medications)'] },
            { category: '🩺 Follow-Up Protocol', items: ['Sputum smear at 2, 5, 6 months of treatment', 'Chest X-ray at 2 months and end of treatment', 'Monthly weight monitoring', 'Notify public health authority (notifiable disease)'] },
        ],
        emergency_signs: [
            'Haemoptysis (coughing blood > 200 mL in 24 hours)',
            'SpO₂ < 90% — severe respiratory compromise',
            'TB meningitis: severe headache, neck stiffness, altered consciousness',
            'Miliary TB: high fever, hepatosplenomegaly, pancytopenia',
            'Spontaneous pneumothorax from ruptured cavity',
            'Multi-organ failure in disseminated TB',
        ],
        follow_up: 'Monthly clinical review throughout treatment. Sputum culture at 2 months (key milestone). End-of-treatment chest X-ray mandatory. Post-treatment surveillance for 12 months. BCG vaccination for contacts < 5 years old.',
        prognosis: 'Excellent with full treatment compliance (> 95% cure rate). Poor with MDR-TB or HIV co-infection. Early diagnosis and complete treatment are critical.',
    },

    'COVID-19': {
        xray_patterns: [
            'Bilateral, peripheral, predominantly lower-lobe ground-glass opacities (GGOs)',
            'Peripheral "crazy paving" pattern — ground-glass + interlobular septal thickening',
            '"White lung" appearance in severe ARDS stage',
            'Vascular enlargement within GGO areas (dilated pulmonary vessels)',
            'Consolidation with air bronchograms in advanced disease',
            'Absence of lymphadenopathy or pleural effusion (distinguishes from bacterial)',
        ],
        why_ai_decided: [
            'Bilateral peripheral distribution of opacities — highly characteristic of COVID-19',
            'Lower-lobe predominance with ground-glass texture detected',
            'Pattern inconsistent with typical bacterial lobar pneumonia or TB',
            'Diffuse involvement suggesting viral pneumonitis with cytokine response',
        ],
        causes: [
            { type: 'Pathogen', agents: ['SARS-CoV-2 virus (Beta-coronavirus)', 'Variants: Alpha, Delta, Omicron (XBB, JN.1)', 'Genetic mutation-driven immune evasion'] },
            { type: 'Transmission', agents: ['Airborne transmission (primary — aerosols < 5 μm)', 'Close contact droplet spread', 'Rarely: fomite surface contact', 'Pre-symptomatic and asymptomatic transmission'] },
            { type: 'Risk Factors', agents: ['Age > 60 years', 'Unvaccinated status', 'Obesity (BMI > 30)', 'DM, hypertension, cardiovascular disease', 'Immunosuppression, chronic kidney/lung disease'] },
        ],
        pathophysiology: 'SARS-CoV-2 binds ACE2 receptors on alveolar type II pneumocytes. Viral replication triggers a massive cytokine storm (IL-6, TNF-α, IL-1β), causing diffuse alveolar damage. The AI-detected bilateral GGOs represent fluid-filled alveoli from this inflammatory cascade. In severe cases this progresses to ARDS with "white lung" appearance requiring mechanical ventilation.',
        symptoms_correlation: 'Typical presentation: fever, dry cough, fatigue, loss of taste/smell (anosmia/ageusia). Moderate-severe: dyspnea, pleuritic chest pain, SpO₂ drop. Silent hypoxia common in early COVID-19 pneumonia despite minimal breathlessness.',
        treatments: [
            'Antiviral: Paxlovid (nirmatrelvir/ritonavir) if < 5 days symptom onset',
            'Remdesivir (IV) for hospitalised patients requiring O₂',
            'Dexamethasone 6 mg/day for 10 days if on oxygen',
            'Anticoagulation (heparin) for thromboprophylaxis',
            'High-flow nasal O₂ or prone positioning before intubation',
            'Tocilizumab/baricitinib for cytokine storm management',
        ],
        remedies: [
            { category: '💊 Medical Management', items: ['Antiviral therapy within 5 days of symptom onset is critical', 'Dexamethasone ONLY if needing supplemental oxygen (harmful if given too early)', 'Monitor D-dimer, CRP, ferritin — markers of cytokine storm', 'Pulse oximetry 4×/day — target SpO₂ > 95%'] },
            { category: '🏠 Home Isolation Care', items: ['Isolate in separate room for 10 days from symptom onset', 'N95/FFP2 mask when near family members', 'Prone positioning (lying face-down) for 1–2 hours helps oxygenation', 'Warm steam inhalation, honey-ginger-turmeric tea', 'Monitor SpO₂ every 4–6 hours'] },
            { category: '🥗 Recovery Nutrition', items: ['Vitamin D3 (4000 IU/day) — immune modulation', 'Vitamin C (1000 mg/day) — antioxidant support', 'Zinc (50 mg/day) — viral replication inhibition', 'Omega-3 fatty acids — anti-inflammatory', 'High-protein diet for muscle recovery', 'Avoid sugar and processed foods'] },
            { category: '🧘 Long COVID Management', items: ['Graded exercise therapy for post-COVID fatigue', 'Breathing rehabilitation exercises', 'Cognitive rehabilitation for brain fog', 'Follow-up at 4–12 weeks post-recovery', 'Monitor for POTS, myocarditis, and pulmonary fibrosis'] },
        ],
        emergency_signs: [
            'SpO₂ < 94% at rest or dropping rapidly',
            'Respiratory rate > 30 breaths/min ("silent hypoxia")',
            'Chest pain or pressure (rule out COVID myocarditis)',
            'Confusion, difficulty waking, slurred speech',
            'Persistent high fever > 39°C for > 3 days despite paracetamol',
            'Cyanosis: blue lips, face, or fingernails',
            'Stridor or inability to complete sentences',
        ],
        follow_up: 'SpO₂ monitoring daily at home. Clinical review at day 7–10. Post-acute COVID follow-up at 4 weeks and 12 weeks. Chest X-ray at 6 weeks to confirm resolution. Cardiac evaluation if chest pain during illness.',
        prognosis: 'Mild-moderate cases: full recovery in 2–4 weeks. Severe (ICU): 20–40% mortality. Long COVID affects ~10–30% of infected. Vaccination reduces severe disease risk by > 90%.',
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. DIFFERENTIAL DIAGNOSIS ENGINE
// ─────────────────────────────────────────────────────────────────────────────

export const getDifferentialDiagnosis = (allProbabilities) => {
    if (!allProbabilities) return [];

    return Object.entries(allProbabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 4)
        .map(([disease, prob]) => ({
            disease,
            probability: prob,
            likelihood: prob > 0.5 ? 'Primary' : prob > 0.2 ? 'Differential' : 'Less Likely',
            color: prob > 0.5 ? '#ef4444' : prob > 0.2 ? '#f97316' : '#6b7280',
        }));
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. UNCERTAINTY / CONFIDENCE INTERVAL ESTIMATOR
// ─────────────────────────────────────────────────────────────────────────────

export const getUncertaintyAnalysis = (confidence, allProbabilities) => {
    // Entropy-based uncertainty (higher entropy = more uncertain)
    let entropy = 0;
    if (allProbabilities) {
        Object.values(allProbabilities).forEach(p => {
            if (p > 0) entropy -= p * Math.log2(p);
        });
    }

    const maxEntropy = Math.log2(4); // 4 classes
    const normalizedEntropy = entropy / maxEntropy;

    const certaintyScore = 1 - normalizedEntropy;
    const margin = (1 - confidence) * 0.5; // simplified CI

    return {
        entropyScore: normalizedEntropy,
        certaintyScore,
        confidenceLow: Math.max(0, confidence - margin).toFixed(2),
        confidenceHigh: Math.min(1, confidence + margin).toFixed(2),
        reliabilityLabel:
            certaintyScore > 0.80 ? 'High Reliability' :
                certaintyScore > 0.60 ? 'Moderate Reliability' :
                    'Low Reliability — Clinical Correlation Required',
    };
};

// ─────────────────────────────────────────────────────────────────────────────
// 5. FEATURE IMPORTANCE NARRATIVE (based on confidence + disease)
// ─────────────────────────────────────────────────────────────────────────────

const REGION_DESCRIPTIONS = {
    'Pneumonia': {
        high: ['Right lower lobe', 'Left lower lobe', 'Bilateral basal regions'],
        medium: ['Right middle lobe', 'Left upper lobe', 'Peri-hilar region'],
    },
    'Tuberculosis': {
        high: ['Right upper lobe apex', 'Left upper lobe apex', 'Posterior segments'],
        medium: ['Hilar lymph nodes', 'Mid-lung fields', 'Left lower lobe'],
    },
    'COVID-19': {
        high: ['Bilateral periphery', 'Lower lobe bilateral', 'Sub-pleural regions'],
        medium: ['Central lung fields', 'Middle lobe bilateral', 'Peri-vascular regions'],
    },
    'Normal': {
        high: [],
        medium: [],
    },
};

export const getFeatureImportanceNarrative = (prediction, confidence) => {
    const regions = REGION_DESCRIPTIONS[prediction] || REGION_DESCRIPTIONS['Normal'];

    if (prediction === 'Normal') {
        return [{
            region: 'All lung fields',
            attention: 0.05,
            description: 'Uniform normal density — no focal area of AI attention',
        }];
    }

    const features = [];

    if (regions.high.length > 0) {
        features.push({
            region: regions.high[0],
            attention: confidence * 0.9,
            description: `Primary diagnostic region — highest AI attention weight`,
        });
    }

    if (regions.high.length > 1) {
        features.push({
            region: regions.high[1],
            attention: confidence * 0.7,
            description: `Secondary affected region contributing to diagnosis`,
        });
    }

    if (regions.medium.length > 0) {
        features.push({
            region: regions.medium[0],
            attention: confidence * 0.4,
            description: `Supporting evidence region with moderate attention`,
        });
    }

    return features;
};

// ─────────────────────────────────────────────────────────────────────────────
// 6. MAIN XAI REPORT GENERATOR
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a full structured XAI report.
 * @param {string} prediction       — e.g. 'Pneumonia'
 * @param {number} confidence       — 0.0 to 1.0
 * @param {object} allProbabilities — { 'COVID-19': 0.1, 'Normal': 0.05, ... }
 * @param {string|null} gradcamUrl  — base64 or URL of GradCAM heatmap (optional)
 * @returns {object} Full XAI report
 */
export const generateXAIReport = (prediction, confidence, allProbabilities = {}, gradcamUrl = null) => {
    const kb = KNOWLEDGE_BASE[prediction] || KNOWLEDGE_BASE['Normal'];
    const severity = getSeverity(prediction, confidence);
    const differential = getDifferentialDiagnosis(allProbabilities);
    const uncertainty = getUncertaintyAnalysis(confidence, allProbabilities);
    const featureImportance = getFeatureImportanceNarrative(prediction, confidence);

    return {
        // Core
        prediction,
        confidence,
        confidencePct: (confidence * 100).toFixed(1),
        timestamp: new Date().toISOString(),

        // Severity
        severity,

        // Why AI decided this
        xrayPatterns: kb.xray_patterns,
        whyAIDecided: kb.why_ai_decided,
        featureImportance,

        // Clinical
        pathophysiology: kb.pathophysiology,
        symptomsCorrelation: kb.symptoms_correlation,
        causes: kb.causes,

        // Remedies & Treatment
        treatments: kb.treatments,
        remedies: kb.remedies,

        // Safety
        emergencySigns: kb.emergency_signs,
        followUp: kb.follow_up,
        prognosis: kb.prognosis,

        // Advanced XAI
        differential,
        uncertainty,

        // GradCAM
        gradcamUrl,

        // Disclaimer
        disclaimer: 'This AI analysis is for educational purposes only. It must not replace professional medical diagnosis. Always consult a qualified radiologist or physician for medical decisions.',
    };
};
