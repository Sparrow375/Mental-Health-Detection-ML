import React, { useEffect, useState } from 'react';
import { collection, getDocs, query, orderBy } from 'firebase/firestore';
import { db } from '../firebase/config';
import { useNavigate } from 'react-router-dom';
import { Search, FileText, AlertTriangle, ShieldCheck, Clock, EyeOff } from 'lucide-react';
import { usePrivacy } from '../context/PrivacyContext';

interface Patient {
  id: string;
  email: string;
  patient_id: string;
  status: string;
  onboarding_date: number;
  latest_analysis?: {
    anomaly_score: number;
    timestamp?: number;
    features?: any;
  };
}

import { getLatestResult } from '../firebase/dataHelper';

export const PatientList: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  const { isAnonymous } = usePrivacy();

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const q = query(collection(db, 'users'));
        const querySnapshot = await getDocs(q);
        
        const dataPromises = querySnapshot.docs.map(async (docSnap) => {
          const userData = docSnap.data();
          const uid = docSnap.id;
          
          const latestResult = await getLatestResult(uid);
          
          let status = 'Collecting';
          let score = 0;
          
          if (latestResult) {
             score = latestResult.anomaly_score || 0;
             status = score >= 0.7 ? 'Flagged' : 'Monitoring';
          }
          
          const onboardingDate = userData.onboardingTimestamp || Date.now();

          return {
            id: uid,
            email: userData.email || userData.name || 'Anonymous User',
            patient_id: `PT-${uid.substring(0, 6).toUpperCase()}`,
            status: status,
            onboarding_date: onboardingDate,
            latest_analysis: latestResult ? { anomaly_score: score } : undefined
          } as Patient;
        });

        const data = await Promise.all(dataPromises);
        setPatients(data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch patients:", err);
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  const filteredPatients = patients.filter(p => 
    p.email?.toLowerCase().includes(searchTerm.toLowerCase()) || 
    p.patient_id?.toLowerCase().includes(searchTerm.toLowerCase())
  ).sort((a, b) => {
    const scoreA = a.latest_analysis?.anomaly_score || (a.status === 'Flagged' ? 0.8 : 0.2);
    const scoreB = b.latest_analysis?.anomaly_score || (b.status === 'Flagged' ? 0.8 : 0.2);
    return scoreB - scoreA;
  });

  const getStatusBadge = (status: string) => {
    if (status === 'Flagged') return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', background: 'rgba(239, 68, 68, 0.15)', color: 'var(--danger)', fontSize: '0.875rem', fontWeight: 500 }}>
        <AlertTriangle size={14} /> Flagged
      </span>
    );
    if (status === 'Collecting') return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', background: 'rgba(245, 158, 11, 0.15)', color: 'var(--warning)', fontSize: '0.875rem', fontWeight: 500 }}>
        <Clock size={14} /> Building Baseline
      </span>
    );
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', fontSize: '0.875rem', fontWeight: 500 }}>
        <ShieldCheck size={14} /> Monitoring
      </span>
    );
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Patient Directory</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Manage and monitor all active devices securely</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div style={{ position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input 
              type="text" 
              placeholder="Search by ID or email..." 
              className="input-field"
              style={{ paddingLeft: '2.5rem', width: '300px' }}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.1)' }}>
              <th style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.875rem' }}>Anonymous ID</th>
              <th style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.875rem' }}>Email / Contact</th>
              <th style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.875rem' }}>System Status</th>
              <th style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.875rem' }}>Onboarding Date</th>
              <th style={{ padding: '1rem 1.5rem', textAlign: 'right', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.875rem' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={5} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading records...</td></tr>
            ) : filteredPatients.length === 0 ? (
              <tr><td colSpan={5} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No patients found.</td></tr>
            ) : (
              filteredPatients.map(patient => (
                <tr 
                  key={patient.id} 
                  style={{ borderBottom: '1px solid var(--border)', transition: 'background var(--transition-fast)' }}
                  onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-card-hover)'}
                  onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '1rem 1.5rem', fontWeight: 500 }}>{patient.patient_id}</td>
                  <td style={{ padding: '1rem 1.5rem', color: isAnonymous ? 'var(--text-muted)' : 'var(--text-secondary)' }}>
                    {isAnonymous ? <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}><EyeOff size={14}/> HIDDEN</span> : patient.email}
                  </td>
                  <td style={{ padding: '1rem 1.5rem' }}>{getStatusBadge(patient.status)}</td>
                  <td style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                    {new Date(patient.onboarding_date).toLocaleDateString()}
                  </td>
                  <td style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>
                    <button 
                      onClick={() => navigate(`/patients/${patient.id}`)}
                      className="btn btn-secondary" 
                      style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }}
                    >
                      <FileText size={16} /> View Data
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
