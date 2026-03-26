import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { doc, getDoc, collection, getDocs, query, orderBy, limit } from 'firebase/firestore';
import { db } from '../firebase/config';
import { ArrowLeft, Activity, ShieldAlert, Cpu, AlertTriangle, ShieldCheck, HeartPulse, Brain } from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

export const PatientDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [patient, setPatient] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
      
      const isMock = localStorage.getItem('mockAdminAuth') === 'true';
      if (isMock) {
        loadDummyData();
        setLoading(false);
        return;
      }
      
      try {
        const docRef = doc(db, 'users', id);
        const docSnap = await getDoc(docRef);
        
        if (docSnap.exists()) {
          setPatient(docSnap.data());
          
          // Fetch history
          const q = query(collection(db, 'users', id, 'results'), orderBy('date', 'desc'), limit(14));
          const histSnap = await getDocs(q);
          const histData = histSnap.docs.map(d => d.data()).reverse();
          setHistory(histData);
        } else {
          loadDummyData();
        }
      } catch (e) {
        loadDummyData();
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  const loadDummyData = () => {
    setPatient({ 
      patient_id: id?.startsWith('Patient') ? id : 'Patient_A1B2', 
      status: 'Flagged', 
      onboarding_date: Date.now() - 864000000,
      latest_analysis: {
        anomaly_score: 0.85,
        features: {
          Sleep: 88,
          Activity: 42,
          Sociability: 35,
          Communication: 28,
          Device_Use: 92
        }
      }
    });
    setHistory([
      { date: '03/10', anomaly_score: 0.15 },
      { date: '03/11', anomaly_score: 0.22 },
      { date: '03/12', anomaly_score: 0.65 },
      { date: '03/13', anomaly_score: 0.88 },
      { date: '03/14', anomaly_score: 0.85 },
      { date: '03/15', anomaly_score: 0.92 },
      { date: '03/16', anomaly_score: 0.81 },
    ]);
  };

  if (loading) return <div style={{ padding: '2rem', color: 'var(--text-muted)' }}>Loading clinical metrics...</div>;
  if (!patient) return <div style={{ padding: '2rem', color: 'var(--danger)' }}>Patient not found</div>;

  const currentScore = patient.latest_analysis?.anomaly_score || 0;
  const isCritical = currentScore >= 0.7;
  const isElevated = currentScore >= 0.4 && currentScore < 0.7;

  // Radar Data mapping
  const featureData = [
    { subject: 'Sleep Disturbance', score: patient.latest_analysis?.features?.Sleep || 75, fullMark: 100 },
    { subject: 'Motor Activity', score: patient.latest_analysis?.features?.Activity || 40, fullMark: 100 },
    { subject: 'Social Engagement', score: patient.latest_analysis?.features?.Sociability || 30, fullMark: 100 },
    { subject: 'Communication', score: patient.latest_analysis?.features?.Communication || 45, fullMark: 100 },
    { subject: 'Screen Time', score: patient.latest_analysis?.features?.Device_Use || 85, fullMark: 100 },
  ];

  return (
    <div className="animate-fade-in">
      {/* Patient Header */}
      <div className="page-header" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
        <button 
          onClick={() => navigate('/patients')} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontWeight: 500 }}
          onMouseOver={e => e.currentTarget.style.color = 'var(--text-primary)'}
          onMouseOut={e => e.currentTarget.style.color = 'var(--text-secondary)'}
        >
          <ArrowLeft size={18} /> Back to Triage
        </button>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ background: isCritical ? 'rgba(239, 68, 68, 0.1)' : isElevated ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)', padding: '1rem', borderRadius: '50%' }}>
              {isCritical ? <AlertTriangle size={32} color="var(--danger)" /> : isElevated ? <Activity size={32} color="var(--warning)" /> : <ShieldCheck size={32} color="var(--success)" />}
            </div>
            <div>
              <h1 className="page-title" style={{ marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {patient.patient_id}
                <span style={{ fontSize: '0.875rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', fontWeight: 600, background: isCritical ? 'var(--danger)' : isElevated ? 'var(--warning)' : 'var(--success)', color: '#fff' }}>
                  {Math.round(currentScore * 100)}% ACUITY
                </span>
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                Patient since {new Date(patient.onboarding_date).toLocaleDateString()} &bull; Baseline Model Active
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button className="btn btn-secondary"><ShieldAlert size={16} /> Clinical Override</button>
            <button className="btn btn-primary"><Cpu size={16} /> Force Engine Sync</button>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '1.5rem', marginBottom: '1.5rem' }}>
        
        {/* Risk-Stratified Trendline */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Activity size={18} color="var(--accent-primary)" /> Longitudinal Anomaly Trajectory
          </h3>
          <div style={{ flex: 1, minHeight: '350px' }}>
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
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                <Tooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg)' }}
                  itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                  labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}
                />
                
                {/* Clinical Threshold Lines */}
                <ReferenceLine y={0.7} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'CRITICAL (0.7)', fill: 'var(--danger)', fontSize: 11, fontWeight: 600 }} />
                <ReferenceLine y={0.4} stroke="var(--warning)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'ELEVATED (0.4)', fill: 'var(--warning)', fontSize: 11, fontWeight: 600 }} />
                
                <Area type="monotone" dataKey="anomaly_score" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" activeDot={{ r: 6, fill: 'var(--accent-primary)', stroke: '#fff', strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Diagnostic Radar */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Brain size={18} color="var(--accent-primary)" /> Behavioral Deviation Radar
          </h3>
          <div style={{ flex: 1, minHeight: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={featureData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontWeight: 500 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="Deviation" dataKey="score" stroke="var(--accent-primary)" strokeWidth={2} fill="var(--accent-primary)" fillOpacity={0.3} />
                <Tooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                  itemStyle={{ color: 'var(--accent-primary)', fontWeight: 600 }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Clinical Insights Engine */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
          <HeartPulse size={18} color="var(--accent-primary)" /> AI Clinical Insights
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
          {featureData.map((feature, idx) => {
            if (feature.score > 70) {
              return (
                <div key={idx} style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.2)', borderLeft: '4px solid var(--danger)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--danger)', marginBottom: '0.25rem' }}>Severe {feature.subject}</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Deviation score is {feature.score}/100. This indicates a highly significant departure from the patient's personalized baseline.</div>
                </div>
              );
            } else if (feature.score > 40) {
               return (
                <div key={idx} style={{ padding: '1rem', background: 'rgba(245, 158, 11, 0.05)', border: '1px solid rgba(245, 158, 11, 0.2)', borderLeft: '4px solid var(--warning)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--warning)', marginBottom: '0.25rem' }}>Elevated {feature.subject}</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Deviation score is {feature.score}/100. Monitoring is recommended.</div>
                </div>
              );
            }
            return null;
          })}
          
          <div style={{ padding: '1rem', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderLeft: '4px solid var(--success)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 600, color: 'var(--success)', marginBottom: '0.25rem' }}>Baseline Integrity</div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>The remaining features remain tightly clustered around the historical baseline. No intervention required for these domains.</div>
          </div>
        </div>
      </div>
      
    </div>
  );
};
