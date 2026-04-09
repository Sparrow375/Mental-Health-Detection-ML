import React, { useEffect, useState } from 'react';
import { collection, getDocs, query, orderBy } from 'firebase/firestore';
import { db } from '../firebase/config';
import { AlertCircle, AlertTriangle, ShieldCheck, Activity, Users, ActivitySquare } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { getLatestResult } from '../firebase/dataHelper';

interface Patient {
  id: string;
  email: string;
  patient_id: string;
  status: string;
  onboarding_date: number;
  latest_analysis?: {
    anomaly_score: number;
    timestamp: number;
  };
}

export const Overview: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPatients = async () => {

      try {
        const q = query(collection(db, 'users'), orderBy('onboarding_date', 'desc'));
        const querySnapshot = await getDocs(q);
        
        const dataPromises = querySnapshot.docs.map(async (docSnap) => {
          const userData = docSnap.data();
          const uid = docSnap.id;
          
          const latestResult = await getLatestResult(uid);
          
          let status = 'Collecting';
          let score = 0;
          let timestamp = Date.now();
          
          if (latestResult) {
            score = latestResult.anomaly_score || 0;
            status = score >= 0.7 ? 'Flagged' : 'Monitoring';
            timestamp = new Date(latestResult.date).getTime() || Date.now();
          }
          
          const onboardingDate = userData.onboardingTimestamp || userData.onboarding_date || Date.now();

          return {
            id: uid,
            email: userData.email || userData.name || 'Anonymous User',
            patient_id: `PT-${uid.substring(0, 6).toUpperCase()}`,
            status: status,
            onboarding_date: onboardingDate,
            latest_analysis: latestResult ? { anomaly_score: score, timestamp } : undefined
          } as Patient;
        });

        const data = await Promise.all(dataPromises);
        setPatients(data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch triage overview:", err);
        setLoading(false);
      }
    };
    
    fetchPatients();
  }, []);

  const getRiskLevel = (p: Patient) => {
    const score = p.latest_analysis?.anomaly_score || (p.status === 'Flagged' ? 0.8 : 0.2);
    if (score >= 0.7) return 'critical';
    if (score >= 0.4) return 'elevated';
    return 'stable';
  };

  const critical = patients.filter(p => getRiskLevel(p) === 'critical');
  const elevated = patients.filter(p => getRiskLevel(p) === 'elevated');
  const stable = patients.filter(p => getRiskLevel(p) === 'stable');

  const avgScore = patients.length > 0
    ? patients.reduce((acc, p) => acc + (p.latest_analysis?.anomaly_score || 0.2), 0) / patients.length
    : 0;

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh', flexDirection: 'column', gap: '1rem', color: 'var(--text-muted)' }}>
        <Activity size={32} className="animate-spin" />
        <p>Loading clinical dashboard...</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Triage Center</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Live system vitals and patient risk stratification</p>
        </div>
      </div>
      
      {/* System Vitals */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '3rem' }}>
        <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: '4px solid var(--danger)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Critical Attention</h3>
            <div style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '0.5rem', borderRadius: '50%' }}>
              <AlertCircle size={20} color="var(--danger)" />
            </div>
          </div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--danger)', lineHeight: 1 }}>{critical.length}</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.5rem' }}>Patients requiring review</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Network Acuity</h3>
            <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '0.5rem', borderRadius: '50%' }}>
              <ActivitySquare size={20} color="var(--accent-primary)" />
            </div>
          </div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--accent-primary)', lineHeight: 1 }}>
            {(avgScore * 100).toFixed(1)}<span style={{ fontSize: '1.25rem', color: 'var(--text-muted)' }}>%</span>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.5rem' }}>Average anomaly severity</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: '4px solid var(--success)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Active Monitors</h3>
            <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '0.5rem', borderRadius: '50%' }}>
              <Users size={20} color="var(--success)" />
            </div>
          </div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>{patients.length}</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.5rem' }}>Total enrolled patients</div>
        </div>
      </div>

      {/* Patient Triage Boards */}
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>Active Patient Triage</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
        
        {/* CRITICAL */}
        <div className="glass-panel" style={{ padding: '1.5rem', background: 'var(--bg-card)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--danger)', marginBottom: '1.5rem' }}>
            <AlertCircle size={20} />
            <h3 style={{ fontWeight: 600 }}>Critical Priority ({critical.length})</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {critical.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No critical patients.</p> : null}
            {critical.map(p => (
              <div key={p.id} onClick={() => navigate(`/dashboard/patients/${p.id}`)} style={{ padding: '1rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(239, 68, 68, 0.2)', borderLeft: '3px solid var(--danger)', cursor: 'pointer', transition: 'all 0.2s ease' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600 }}>{p.patient_id}</span>
                  <span style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '1.1rem' }}>{Math.round((p.latest_analysis?.anomaly_score || 0.8) * 100)}%</span>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>High Deviation Detected</div>
              </div>
            ))}
          </div>
        </div>

        {/* ELEVATED */}
        <div className="glass-panel" style={{ padding: '1.5rem', background: 'var(--bg-card)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--warning)', marginBottom: '1.5rem' }}>
            <AlertTriangle size={20} />
            <h3 style={{ fontWeight: 600 }}>Elevated Risk ({elevated.length})</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {elevated.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No elevated patients.</p> : null}
            {elevated.map(p => (
              <div key={p.id} onClick={() => navigate(`/dashboard/patients/${p.id}`)} style={{ padding: '1rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(245, 158, 11, 0.2)', borderLeft: '3px solid var(--warning)', cursor: 'pointer', transition: 'all 0.2s ease' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600 }}>{p.patient_id}</span>
                  <span style={{ color: 'var(--warning)', fontWeight: 700, fontSize: '1.1rem' }}>{Math.round((p.latest_analysis?.anomaly_score || 0.5) * 100)}%</span>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Monitoring Required</div>
              </div>
            ))}
          </div>
        </div>

        {/* STABLE */}
        <div className="glass-panel" style={{ padding: '1.5rem', background: 'var(--bg-card)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--success)', marginBottom: '1.5rem' }}>
            <ShieldCheck size={20} />
            <h3 style={{ fontWeight: 600 }}>Stable/Baselining ({stable.length})</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {stable.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No stable patients.</p> : null}
            {stable.map(p => (
              <div key={p.id} onClick={() => navigate(`/dashboard/patients/${p.id}`)} style={{ padding: '1rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.2)', borderLeft: '3px solid var(--success)', cursor: 'pointer', transition: 'all 0.2s ease' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600 }}>{p.patient_id}</span>
                  <span style={{ color: 'var(--success)', fontWeight: 700, fontSize: '1.1rem' }}>{Math.round((p.latest_analysis?.anomaly_score || 0.1) * 100)}%</span>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Normal Ranges</div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};
