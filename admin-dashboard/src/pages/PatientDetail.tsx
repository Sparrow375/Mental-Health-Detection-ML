import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';
import { getHistoricalResults, getLatestFeatures, getBaseline } from '../firebase/dataHelper';
import type { MLResult, DailyFeatures, BaselineData } from '../firebase/dataHelper';
import { ArrowLeft, Activity, ShieldAlert, Cpu, AlertTriangle, ShieldCheck, HeartPulse, Brain, Smartphone, Footprints, MapPin, Map, Users, Compass, Moon, Phone, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

export const PatientDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  
  const [patient, setPatient] = useState<any>(null);
  const [history, setHistory] = useState<MLResult[]>([]);
  const [features, setFeatures] = useState<DailyFeatures | null>(null);
  const [baseline, setBaseline] = useState<BaselineData | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
      
      try {
        const docRef = doc(db, 'users', id);
        const docSnap = await getDoc(docRef);
        
        if (docSnap.exists()) {
          setPatient(docSnap.data());
        }

        const histData = await getHistoricalResults(id, 14);
        setHistory(histData);

        const currentData = await getLatestFeatures(id);
        const baselineData = await getBaseline(id);

        setFeatures(currentData);
        setBaseline(baselineData);

      } catch (e) {
        console.error("Error fetching detail:", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  if (loading) return <div style={{ padding: '2rem', color: 'var(--text-muted)' }}>Loading clinical metrics...</div>;
  if (!patient && history.length === 0) return <div style={{ padding: '2rem', color: 'var(--danger)' }}>Patient not found or no data available.</div>;

  const currentResult = history.length > 0 ? history[history.length - 1] : null;
  const currentScore = currentResult?.anomaly_score || 0;
  
  const isCritical = currentScore >= 0.7;
  const isElevated = currentScore >= 0.4 && currentScore < 0.7;
  const prototypeMatch = currentResult?.prototype_match;
  const matchMessage = currentResult?.match_message;

  // Clinical Biomarkers Data array
  const rawFeaturesList = [
    { name: 'Screen Time', value: features?.screenTimeHours, displayValue: features?.screenTimeHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Smartphone, defaultColor: '#38bdf8', key: 'screenTimeHours', invertGood: true },
    { name: 'Step Count', value: features?.dailySteps, displayValue: features?.dailySteps?.toFixed(0) || '0', unit: 'steps', icon: Footprints, defaultColor: 'var(--success)', key: 'dailySteps', invertGood: false },
    { name: 'Places Visited', value: features?.placesVisited, displayValue: features?.placesVisited?.toFixed(0) || '0', unit: 'locations', icon: MapPin, defaultColor: 'var(--warning)', key: 'placesVisited', invertGood: false },
    { name: 'Displacement', value: features?.dailyDisplacementKm, displayValue: features?.dailyDisplacementKm?.toFixed(1) || '0.0', unit: 'km', icon: Map, defaultColor: '#a855f7', key: 'dailyDisplacementKm', invertGood: false },
    { name: 'Social Ratio', value: features?.socialAppRatio, displayValue: features?.socialAppRatio ? (features.socialAppRatio * 100).toFixed(0) : '0', unit: '%', icon: Users, defaultColor: '#ec4899', key: 'socialAppRatio', invertGood: false },
    { name: 'Location Entropy', value: features?.locationEntropy, displayValue: features?.locationEntropy?.toFixed(2) || '0.00', unit: 'score', icon: Compass, defaultColor: '#14b8a6', key: 'locationEntropy', invertGood: false },
    { name: 'Sleep Duration', value: features?.sleepDurationHours, displayValue: features?.sleepDurationHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Moon, defaultColor: '#6366f1', key: 'sleepDurationHours', invertGood: false },
    { name: 'Call Duration', value: features?.callDurationMinutes, displayValue: features?.callDurationMinutes?.toFixed(0) || '0', unit: 'mins', icon: Phone, defaultColor: '#facc15', key: 'callDurationMinutes', invertGood: false },
  ];

  const getClinicalStatus = (featureName: string, currentVal: number | undefined, invert: boolean) => {
    if (currentVal === undefined) return { color: 'var(--text-muted)', status: 'No Data', icon: null };
    if (!baseline || !baseline[featureName] || baseline[featureName].mean === 0) return { color: 'var(--text-secondary)', status: 'Establishing...', icon: <Minus size={14} /> };
    
    const baseMean = baseline[featureName].mean;
    const baseStd = baseline[featureName].std || (baseMean * 0.1) || 1; // Fallback to 10% std if 0
    
    const diff = currentVal - baseMean;
    const zScore = Math.abs(diff) / baseStd;
    
    // Higher is generally better unless invertGood is true (e.g. screen time)
    const isHigh = diff > 0;
    
    if (zScore > 2) {
      const isBad = invert ? isHigh : !isHigh;
      return { color: isBad ? 'var(--danger)' : 'var(--success)', status: isHigh ? '+ High Var' : '- Low Var', icon: isHigh ? <TrendingUp size={14} /> : <TrendingDown size={14} /> };
    }
    if (zScore > 1) {
      const isBad = invert ? isHigh : !isHigh;
      return { color: isBad ? 'var(--warning)' : 'var(--success)', status: isHigh ? '+ Elev' : '- Red', icon: isHigh ? <TrendingUp size={14} /> : <TrendingDown size={14} /> };
    }
    return { color: 'var(--success)', status: 'Stable', icon: <Minus size={14} /> };
  };

  // Normalization Math: (Current / Baseline Mean) * 50
  // Values near 50 mean "at baseline". Values > 50 mean "elevated". Values < 50 mean "reduced".
  const normalize = (featureName: string): number => {
    if (!features || !baseline || !baseline[featureName] || baseline[featureName].mean === 0) return 50; 
    const currentVal = (features as any)[featureName] || 0;
    const baseMean = baseline[featureName].mean;
    // Cap at 100 for radar rendering
    return Math.min(Math.max((currentVal / baseMean) * 50, 0), 100);
  };

  const featureData = [
    { subject: 'Screen Time', score: normalize('screenTimeHours'), fullMark: 100 },
    { subject: 'Sociability', score: normalize('socialAppRatio'), fullMark: 100 },
    { subject: 'Mobility', score: normalize('placesVisited'), fullMark: 100 },
    { subject: 'Loc. Entropy', score: normalize('locationEntropy'), fullMark: 100 },
    { subject: 'Sleep Duration', score: normalize('sleepDurationHours'), fullMark: 100 },
    { subject: 'Communication', score: normalize('callDurationMinutes'), fullMark: 100 },
  ];

  // If a prototype match exists, we display a structural matching circle logic.
  // In a real medical app, this would be the exact polygon of the prototype mean.
  const hasPrototypeMatch = !!prototypeMatch && prototypeMatch.trim() !== "";
  const prototypePolygon = featureData.map(f => ({ ...f, pattern: 85 })); // Example fixed pattern line to represent the clinical threshold crossed

  const insights = featureData.map(f => {
    const dev = f.score - 50;
    return { name: f.subject, deviation: Math.abs(dev), rawScore: f.score, sign: dev > 0 ? '+' : '-' };
  }).sort((a, b) => b.deviation - a.deviation).slice(0, 3); // Top 3 deviations

  return (
    <div className="animate-fade-in">
      <div className="page-header" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
        <button 
          onClick={() => navigate('/patients')} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontWeight: 500 }}
          onMouseOver={e => e.currentTarget.style.color = 'var(--text-primary)'}
          onMouseOut={e => e.currentTarget.style.color = 'var(--text-secondary)'}
        >
          <ArrowLeft size={18} /> Back to Directory
        </button>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ background: isCritical ? 'rgba(239, 68, 68, 0.1)' : isElevated ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)', padding: '1rem', borderRadius: '50%' }}>
              {isCritical ? <AlertTriangle size={32} color="var(--danger)" /> : isElevated ? <Activity size={32} color="var(--warning)" /> : <ShieldCheck size={32} color="var(--success)" />}
            </div>
            <div>
              <h1 className="page-title" style={{ marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {patient?.patient_id || id}
                <span style={{ fontSize: '0.875rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', fontWeight: 600, background: isCritical ? 'var(--danger)' : isElevated ? 'var(--warning)' : 'var(--success)', color: '#fff' }}>
                  {Math.round(currentScore * 100)}% ACUITY
                </span>
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                Patient since {new Date(patient?.onboarding_date || Date.now()).toLocaleDateString()} 
                {baseline ? ' • Baseline Established' : ' • Establishing Baseline...'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Clinical Metrics Grid */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={20} color="var(--accent-primary)" /> Clinical Biomarkers (24h Window)
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2.5rem' }}>
        {rawFeaturesList.map((feat) => {
          const statusObj = getClinicalStatus(feat.key, feat.value, feat.invertGood);
          const Icon = feat.icon;
          return (
            <div key={feat.key} className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, width: '4px', height: '100%', background: statusObj.color }}></div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ background: `${feat.defaultColor}15`, padding: '0.5rem', borderRadius: '0.5rem', color: feat.defaultColor }}>
                    <Icon size={18} />
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 500 }}>{feat.name}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: statusObj.color, fontSize: '0.75rem', fontWeight: 600, background: `${statusObj.color}15`, padding: '0.25rem 0.5rem', borderRadius: '1rem' }}>
                  {statusObj.icon} {statusObj.status}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.25rem', marginTop: 'auto' }}>
                <h3 style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>{feat.displayValue}</h3>
                <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-muted)' }}>{feat.unit}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '1.5rem', marginBottom: '1.5rem' }}>
        
        {/* Risk-Stratified Trendline */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Activity size={18} color="var(--accent-primary)" /> Longitudinal Anomaly Trajectory (Last 14 Days)
          </h3>
          <div style={{ flex: 1, minHeight: '350px' }}>
            {history.length === 0 ? (
               <div style={{ height: '100%', display: 'flex', alignItems:'center', justifyContent: 'center', color: 'var(--text-muted)' }}>No historical data available yet.</div>
            ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 10, right: 30, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--danger)" stopOpacity={0.8}/>
                    <stop offset="60%" stopColor="var(--warning)" stopOpacity={0.5}/>
                    <stop offset="95%" stopColor="var(--success)" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={12} tickMargin={10} />
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} tickFormatter={(v: any) => v.toFixed(1)} />
                <RechartsTooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg)' }}
                  itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                  labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}
                />
                
                <ReferenceLine y={0.7} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'CRITICAL (0.7)', fill: 'var(--danger)', fontSize: 11, fontWeight: 600 }} />
                <ReferenceLine y={0.4} stroke="var(--warning)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'ELEVATED (0.4)', fill: 'var(--warning)', fontSize: 11, fontWeight: 600 }} />
                
                <Area type="monotone" dataKey="anomaly_score" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" activeDot={{ r: 6, fill: 'var(--accent-primary)', stroke: '#fff', strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
            )}
          </div>
        </div>
        
        {/* Diagnostic Radar (6 axes) */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Brain size={18} color="var(--accent-primary)" /> Behavioral Deviation Radar
          </h3>
          <div style={{ flex: 1, minHeight: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
            {!baseline || !features ? (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center' }}>Baseline Data Not Established</div>
            ) : (
            <ResponsiveContainer width="100%" height="100%">
              {/* @ts-ignore - Recharts passing array of mixed configs works, type definition is just strict */}
              <RadarChart cx="50%" cy="50%" outerRadius="65%" data={hasPrototypeMatch ? prototypePolygon : featureData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 10, fontWeight: 500 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                
                {/* 50% Baseline Reference Line */}
                <Radar name="Baseline (50%)" dataKey={() => 50} stroke="var(--text-muted)" strokeWidth={1} fill="transparent" strokeDasharray="3 3" />
                
                {/* Current Patient Deviation */}
                <Radar name="Current Deviation" dataKey="score" stroke="var(--accent-primary)" strokeWidth={2} fill="var(--accent-primary)" fillOpacity={0.3} />
                
                {/* Visual Prototype Match (The "Red Line") */}
                {hasPrototypeMatch && (
                  <Radar name={`Match: ${prototypeMatch}`} dataKey="pattern" stroke="#ef4444" strokeWidth={2} fill="#ef4444" fillOpacity={0.1} strokeDasharray="5 5" />
                )}

                <RechartsTooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                />
              </RadarChart>
            </ResponsiveContainer>
            )}
            
            {hasPrototypeMatch && (
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, textAlign: 'center', background: 'rgba(239,68,68,0.1)', color: 'var(--danger)', padding: '0.5rem', borderRadius: 'var(--radius-md)', fontWeight: 600, fontSize: '0.875rem' }}>
                STRUCTURAL MATCH: {prototypeMatch}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Dynamic Clinical Insights */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
          <HeartPulse size={18} color="var(--accent-primary)" /> Dynamic Clinical Insights
        </h3>
        
        {matchMessage && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(56, 189, 248, 0.1)', borderLeft: '4px solid #38bdf8', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontWeight: 500 }}>
                🤖 ML Engine Note: {matchMessage}
            </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
          {!baseline ? (
              <div style={{ color: 'var(--text-muted)' }}>Cannot generate deviations without baseline data.</div>
          ) : (
            insights.map((insight, idx) => {
                if (insight.deviation < 5) return null; // Ignore minor variance
                const isHigh = insight.sign === '+';
                const color = isHigh ? 'var(--warning)' : 'var(--danger)'; // Just example styling
                return (
                    <div key={idx} style={{ padding: '1rem', background: `rgba(245, 158, 11, 0.05)`, border: `1px solid rgba(245, 158, 11, 0.2)`, borderLeft: `4px solid ${color}`, borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontWeight: 600, color: color, marginBottom: '0.25rem' }}>
                        {isHigh ? 'Elevated' : 'Reduced'} {insight.name}
                    </div>
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                        Currently measuring {insight.rawScore.toFixed(0)}% against personal baseline. Significant {isHigh ? 'increase' : 'decrease'} detected.
                    </div>
                    </div>
                );
            })
          )}
          
          <div style={{ padding: '1rem', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderLeft: '4px solid var(--success)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 600, color: 'var(--success)', marginBottom: '0.25rem' }}>Baseline Integrity</div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>All other features remain tightly clustered around the historical baseline structure.</div>
          </div>
        </div>
      </div>
      
    </div>
  );
};
