import React from 'react';
import { Routes, Route, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { auth } from '../firebase/config';
import { signOut } from 'firebase/auth';
import { Activity, Users, Settings as SettingsIcon, LogOut, BarChart3, Eye, EyeOff } from 'lucide-react';
import { PatientList } from './PatientList';
import { PatientDetail } from './PatientDetail';
import { Reports } from './Reports';
import { Overview } from './Overview';
import { UserDashboard } from './UserDashboard';
import { Settings } from './Settings';
import { usePrivacy } from '../context/PrivacyContext';

export const Dashboard: React.FC = () => {
  const { isAnonymous, togglePrivacy } = usePrivacy();
  const role = auth.currentUser?.email?.includes('admin') ? 'admin' : 'user';
  const isAdmin = role === 'admin';

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (e) {
      console.warn("Firebase signout failed", e);
    }
    window.location.href = '/login';
  };

  // Helper for NavLink styles
  const navStyle = ({ isActive }: { isActive: boolean }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    padding: '0.75rem 1rem',
    borderRadius: 'var(--radius-md)',
    color: isActive ? '#fff' : 'var(--text-secondary)',
    background: isActive ? 'var(--accent-primary)' : 'transparent',
    textDecoration: 'none',
    transition: 'all 0.2s ease',
    fontWeight: isActive ? 500 : 400
  });

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div style={{ padding: '1.5rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--accent-primary)' }}>
            <Activity size={24} />
            <span style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)' }}>
              {isAdmin ? 'MHealth Platform' : 'Caregiver Portal'}
            </span>
          </div>
        </div>
        
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1, padding: '0 1rem' }}>
          <NavLink to="/" style={navStyle}>
            <Activity size={20} /> Overview
          </NavLink>
          {isAdmin && (
            <>
              <NavLink to="/patients" style={navStyle}>
                <Users size={20} /> Patients
              </NavLink>
              <NavLink to="/reports" style={navStyle}>
                <BarChart3 size={20} /> Reports
              </NavLink>
            </>
          )}
          <NavLink to="/settings" style={navStyle}>
            <SettingsIcon size={20} /> Settings
          </NavLink>
        </nav>

        <div style={{ marginTop: 'auto', paddingTop: '2rem', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '0 1rem 1rem 1rem' }}>
          <button 
            onClick={togglePrivacy} 
            style={{ 
              display: 'flex', alignItems: 'center', gap: '0.75rem', width: '100%', 
              padding: '0.75rem 1rem', color: isAnonymous ? 'var(--success)' : 'var(--warning)', 
              background: 'var(--bg-card-hover)', cursor: 'pointer', borderRadius: 'var(--radius-md)', 
              border: '1px solid var(--border)', transition: 'all 0.2s' 
            }}
          >
            {isAnonymous ? <EyeOff size={20} /> : <Eye size={20} />} {isAnonymous ? 'Privacy: ON' : 'Privacy: OFF'}
          </button>
          
          <button 
            onClick={handleLogout} 
            style={{ 
              display: 'flex', alignItems: 'center', gap: '0.75rem', width: '100%', 
              padding: '0.75rem 1rem', color: 'var(--danger)', background: 'transparent', 
              border: '1px solid var(--danger)', cursor: 'pointer', borderRadius: 'var(--radius-md)', 
              transition: 'all 0.2s' 
            }}
            onMouseOver={(e) => { e.currentTarget.style.background = 'var(--danger)'; e.currentTarget.style.color = '#fff'; }}
            onMouseOut={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--danger)'; }}
          >
            <LogOut size={20} /> Logout
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={isAdmin ? <Overview /> : <UserDashboard />} />
          {isAdmin && (
            <>
              <Route path="/patients" element={<PatientList />} />
              <Route path="/patients/:id" element={<PatientDetail />} />
              <Route path="/reports" element={<Reports />} />
            </>
          )}
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
};
