import React, { useState } from 'react';

const XAIReportCard = ({ report }) => {
    const [activeTab, setActiveTab] = useState('overview');

    if (!report) return null;

    const {
        prediction, confidencePct, severity,
        xrayPatterns, whyAIDecided, featureImportance,
        pathophysiology, symptomsCorrelation, causes,
        treatments, remedies, emergencySigns, followUp, prognosis,
        differential, uncertainty, gradcamUrl, disclaimer,
    } = report;

    const tabs = [
        { id: 'overview', label: '🔍 Overview' },
        { id: 'causes', label: '🧬 Causes' },
        { id: 'treatment', label: '💊 Treatment' },
        { id: 'xai', label: '🤖 AI Explanation' },
        { id: 'emergency', label: '🚨 Emergency Signs' },
    ];

    return (
        <div style={styles.container}>
            {/* ── HEADER ── */}
            <div style={{ ...styles.header, background: `linear-gradient(135deg, ${severity.color}22, ${severity.color}44)`, borderLeft: `5px solid ${severity.color}` }}>
                <div style={styles.headerLeft}>
                    <span style={styles.severityIcon}>{severity.icon}</span>
                    <div>
                        <h2 style={styles.diagnosis}>{prediction}</h2>
                        <p style={styles.confidence}>AI Confidence: <strong>{confidencePct}%</strong></p>
                    </div>
                </div>
                <div style={{ ...styles.severityBadge, background: severity.bg, color: severity.color, border: `1.5px solid ${severity.color}` }}>
                    <span style={styles.severityLabel}>{severity.label} Severity</span>
                </div>
            </div>

            {/* ── CONFIDENCE INTERVAL ── */}
            <div style={styles.uncertaintyBar}>
                <span style={styles.uncertaintyLabel}>📊 {uncertainty.reliabilityLabel}</span>
                <span style={styles.ciText}>CI: {uncertainty.confidenceLow} – {uncertainty.confidenceHigh}</span>
            </div>

            {/* ── TAB BAR ── */}
            <div style={styles.tabBar}>
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        style={{ ...styles.tab, ...(activeTab === tab.id ? { ...styles.tabActive, borderBottomColor: severity.color, color: severity.color } : {}) }}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* ══════════════════════════════════════
          TAB: OVERVIEW
      ══════════════════════════════════════ */}
            {activeTab === 'overview' && (
                <div style={styles.tabContent}>
                    {/* GradCAM */}
                    {gradcamUrl && (
                        <Section title="🔥 GradCAM Heatmap — Where AI Looked" color={severity.color}>
                            <img src={gradcamUrl} alt="GradCAM heatmap" style={styles.gradcam} />
                            <p style={styles.note}>Red/orange areas = regions with highest AI attention for this diagnosis</p>
                        </Section>
                    )}

                    {/* Feature Importance */}
                    <Section title="🎯 AI Attention Regions" color={severity.color}>
                        {featureImportance.map((f, i) => (
                            <div key={i} style={styles.featureRow}>
                                <div style={styles.featureMeta}>
                                    <span style={styles.featureRegion}>{f.region}</span>
                                    <span style={styles.featureDesc}>{f.description}</span>
                                </div>
                                <div style={styles.attentionBarBg}>
                                    <div style={{ ...styles.attentionBarFill, width: `${Math.min(100, f.attention * 100)}%`, background: severity.color }} />
                                </div>
                                <span style={{ ...styles.attentionPct, color: severity.color }}>{(f.attention * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </Section>

                    {/* Differential */}
                    <Section title="📈 Differential Diagnosis" color={severity.color}>
                        {differential.map((d, i) => (
                            <div key={i} style={styles.diffRow}>
                                <span style={styles.diffDisease}>{d.disease}</span>
                                <span style={{ ...styles.diffLikelihood, color: d.color }}>{d.likelihood}</span>
                                <div style={styles.diffBarBg}>
                                    <div style={{ ...styles.diffBarFill, width: `${d.probability * 100}%`, background: d.color }} />
                                </div>
                                <span style={{ ...styles.diffPct, color: d.color }}>{(d.probability * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </Section>

                    {/* Pathophysiology */}
                    <Section title="🔬 Pathophysiology — Why This Happens" color={severity.color}>
                        <p style={styles.paragraph}>{pathophysiology}</p>
                    </Section>

                    {/* Symptoms */}
                    <Section title="🩺 Expected Symptoms" color={severity.color}>
                        <p style={styles.paragraph}>{symptomsCorrelation}</p>
                    </Section>

                    {/* Prognosis & Follow-up */}
                    <div style={styles.twoCol}>
                        <InfoBox title="📋 Prognosis" color={severity.color}>{prognosis}</InfoBox>
                        <InfoBox title="📅 Follow-Up" color={severity.color}>{followUp}</InfoBox>
                    </div>
                </div>
            )}

            {/* ══════════════════════════════════════
          TAB: CAUSES
      ══════════════════════════════════════ */}
            {activeTab === 'causes' && (
                <div style={styles.tabContent}>
                    {causes.length === 0 ? (
                        <div style={styles.normalMsg}>✅ No disease detected. The chest X-ray appears normal.</div>
                    ) : (
                        causes.map((c, i) => (
                            <Section key={i} title={c.type} color={severity.color}>
                                <ul style={styles.bulletList}>
                                    {c.agents.map((a, j) => <li key={j} style={styles.bullet}>{a}</li>)}
                                </ul>
                            </Section>
                        ))
                    )}
                </div>
            )}

            {/* ══════════════════════════════════════
          TAB: TREATMENT
      ══════════════════════════════════════ */}
            {activeTab === 'treatment' && (
                <div style={styles.tabContent}>
                    <Section title="🏥 Medical Treatments" color={severity.color}>
                        <ul style={styles.bulletList}>
                            {treatments.map((t, i) => <li key={i} style={styles.bullet}>{t}</li>)}
                        </ul>
                    </Section>

                    {remedies.map((rem, i) => (
                        <Section key={i} title={rem.category} color={severity.color}>
                            <ul style={styles.bulletList}>
                                {rem.items.map((item, j) => <li key={j} style={styles.bullet}>{item}</li>)}
                            </ul>
                        </Section>
                    ))}
                </div>
            )}

            {/* ══════════════════════════════════════
          TAB: AI EXPLANATION
      ══════════════════════════════════════ */}
            {activeTab === 'xai' && (
                <div style={styles.tabContent}>
                    <Section title="🩻 X-Ray Patterns Detected" color={severity.color}>
                        {xrayPatterns.map((p, i) => (
                            <div key={i} style={styles.patternRow}>
                                <span style={{ ...styles.patternDot, background: severity.color }} />
                                <span style={styles.patternText}>{p}</span>
                            </div>
                        ))}
                    </Section>

                    <Section title="🤖 Why the AI Made This Decision" color={severity.color}>
                        {whyAIDecided.map((w, i) => (
                            <div key={i} style={styles.patternRow}>
                                <span style={{ ...styles.patternDot, background: '#6366f1' }} />
                                <span style={styles.patternText}>{w}</span>
                            </div>
                        ))}
                    </Section>

                    <Section title="📊 Model Confidence Analysis" color={severity.color}>
                        <div style={styles.uncertaintyDetail}>
                            <MetricRow label="Confidence Score" value={`${confidencePct}%`} color={severity.color} />
                            <MetricRow label="Confidence Interval" value={`${uncertainty.confidenceLow} – ${uncertainty.confidenceHigh}`} color={severity.color} />
                            <MetricRow label="Entropy Score" value={`${(uncertainty.entropyScore * 100).toFixed(1)}%`} color={severity.color} />
                            <MetricRow label="Model Certainty" value={`${(uncertainty.certaintyScore * 100).toFixed(1)}%`} color={severity.color} />
                            <MetricRow label="Reliability" value={uncertainty.reliabilityLabel} color={severity.color} />
                        </div>
                    </Section>
                </div>
            )}

            {/* ══════════════════════════════════════
          TAB: EMERGENCY SIGNS
      ══════════════════════════════════════ */}
            {activeTab === 'emergency' && (
                <div style={styles.tabContent}>
                    {emergencySigns.length === 0 ? (
                        <div style={styles.normalMsg}>✅ No emergency signs associated with this diagnosis. Follow routine care.</div>
                    ) : (
                        <>
                            <div style={styles.emergencyBanner}>
                                ⚠️ Seek <strong>immediate emergency care</strong> if any of these warning signs appear:
                            </div>
                            {emergencySigns.map((sign, i) => (
                                <div key={i} style={styles.emergencyRow}>
                                    <span style={styles.emergencyIcon}>🚨</span>
                                    <span style={styles.emergencyText}>{sign}</span>
                                </div>
                            ))}
                        </>
                    )}
                </div>
            )}

            {/* ── DISCLAIMER ── */}
            <div style={styles.disclaimer}>⚠️ {disclaimer}</div>
        </div>
    );
};

// ── Helper Sub-components ─────────────────────────────────────────────────────

const Section = ({ title, color, children }) => (
    <div style={styles.section}>
        <h3 style={{ ...styles.sectionTitle, color }}>{title}</h3>
        {children}
    </div>
);

const InfoBox = ({ title, color, children }) => (
    <div style={{ ...styles.infoBox, borderTop: `3px solid ${color}` }}>
        <h4 style={{ ...styles.infoBoxTitle, color }}>{title}</h4>
        <p style={styles.paragraph}>{children}</p>
    </div>
);

const MetricRow = ({ label, value, color }) => (
    <div style={styles.metricRow}>
        <span style={styles.metricLabel}>{label}</span>
        <span style={{ ...styles.metricValue, color }}>{value}</span>
    </div>
);

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
    container: {
        background: '#1a1a2e', color: '#e0e0e0',
        borderRadius: 16, overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        marginTop: 24, fontFamily: "'Inter', sans-serif",
    },
    header: {
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '20px 24px', gap: 16,
    },
    headerLeft: { display: 'flex', alignItems: 'center', gap: 16 },
    severityIcon: { fontSize: 48 },
    diagnosis: { margin: 0, fontSize: 24, fontWeight: 700, color: '#fff' },
    confidence: { margin: '4px 0 0', fontSize: 14, color: '#ccc' },
    severityBadge: { padding: '8px 16px', borderRadius: 20, fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap' },
    severityLabel: { letterSpacing: 0.5 },
    uncertaintyBar: {
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 24px', background: '#16213e', fontSize: 12, color: '#94a3b8',
    },
    uncertaintyLabel: { fontWeight: 500 },
    ciText: { color: '#64748b' },
    tabBar: {
        display: 'flex', borderBottom: '1px solid #2d2d4e',
        background: '#16213e', overflowX: 'auto',
    },
    tab: {
        padding: '12px 20px', background: 'none', border: 'none', borderBottom: '3px solid transparent',
        color: '#64748b', cursor: 'pointer', fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap',
        transition: 'all 0.2s',
    },
    tabActive: { color: '#fff', background: '#1a1a2e', fontWeight: 700 },
    tabContent: { padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 },
    section: {
        background: '#16213e', borderRadius: 10, padding: 16,
    },
    sectionTitle: { margin: '0 0 12px', fontSize: 15, fontWeight: 700 },
    paragraph: { margin: 0, fontSize: 14, lineHeight: 1.7, color: '#cbd5e1' },
    twoCol: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 },
    infoBox: { background: '#16213e', borderRadius: 10, padding: 16 },
    infoBoxTitle: { margin: '0 0 8px', fontSize: 14, fontWeight: 700 },
    bulletList: { margin: 0, paddingLeft: 20 },
    bullet: { marginBottom: 6, fontSize: 14, lineHeight: 1.6, color: '#cbd5e1' },
    patternRow: { display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 },
    patternDot: { width: 8, height: 8, borderRadius: '50%', flexShrink: 0, marginTop: 5 },
    patternText: { fontSize: 14, color: '#cbd5e1', lineHeight: 1.6 },
    featureRow: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 },
    featureMeta: { flex: 1, minWidth: 160 },
    featureRegion: { display: 'block', fontSize: 13, fontWeight: 600, color: '#f1f5f9' },
    featureDesc: { display: 'block', fontSize: 11, color: '#64748b' },
    attentionBarBg: { flex: 2, height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden' },
    attentionBarFill: { height: '100%', borderRadius: 4, transition: 'width 0.5s ease' },
    attentionPct: { fontSize: 12, fontWeight: 700, minWidth: 36, textAlign: 'right' },
    diffRow: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 },
    diffDisease: { minWidth: 120, fontSize: 13, color: '#f1f5f9' },
    diffLikelihood: { fontSize: 11, fontWeight: 700, minWidth: 80 },
    diffBarBg: { flex: 1, height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden' },
    diffBarFill: { height: '100%', borderRadius: 4 },
    diffPct: { fontSize: 12, fontWeight: 700, minWidth: 40, textAlign: 'right' },
    gradcam: { width: '100%', maxHeight: 300, objectFit: 'contain', borderRadius: 8 },
    note: { fontSize: 12, color: '#64748b', marginTop: 8, textAlign: 'center' },
    normalMsg: { background: '#052e16', color: '#4ade80', padding: 16, borderRadius: 8, fontSize: 14 },
    emergencyBanner: {
        background: '#450a0a', color: '#fca5a5', padding: 14, borderRadius: 8,
        fontSize: 14, marginBottom: 12, lineHeight: 1.6,
    },
    emergencyRow: { display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 },
    emergencyIcon: { fontSize: 16, flexShrink: 0 },
    emergencyText: { fontSize: 14, color: '#fca5a5', lineHeight: 1.6 },
    uncertaintyDetail: { display: 'flex', flexDirection: 'column', gap: 8 },
    metricRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #1e293b' },
    metricLabel: { fontSize: 13, color: '#94a3b8' },
    metricValue: { fontSize: 13, fontWeight: 700 },
    disclaimer: { padding: '12px 24px', background: '#0f172a', color: '#475569', fontSize: 11, lineHeight: 1.6 },
};

export default XAIReportCard;
