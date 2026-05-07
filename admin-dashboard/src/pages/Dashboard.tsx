import React, { useState, useCallback } from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import { auth } from '../firebase/config';
import { signOut } from 'firebase/auth';
import { Users, Settings as SettingsIcon, LogOut, BarChart3, Eye, EyeOff, Mic, Menu, X } from 'lucide-react';
import { PatientList } from './PatientList';
import { PatientDetail } from './PatientDetail';
import { Reports } from './Reports';
import { Overview } from './Overview';
import { UserDashboard } from './UserDashboard';
import { Settings } from './Settings';
import { usePrivacy } from '../hooks/usePrivacy';
import { VoiceAssessment } from '../components/VoiceAssessment';

export const Dashboard: React.FC = () => {
  const { isAnonymous, togglePrivacy } = usePrivacy();
  const role = auth.currentUser?.email?.includes('admin') ? 'admin' : 'user';
  const isAdmin = role === 'admin';
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (e) {
      console.warn('Firebase signout failed', e);
    }
    window.location.href = '/login';
  };

  return (
    <div className="app-container">
      {/* Top Header Bar */}
      <header className="top-header">
        {/* Mobile hamburger — hidden on desktop via CSS */}
        <button
          id="mobile-menu-toggle"
          className="hamburger-btn"
          onClick={() => setSidebarOpen(prev => !prev)}
          aria-label="Toggle navigation"
        >
          {sidebarOpen ? <X size={22} /> : <Menu size={22} />}
        </button>

        <div className="header-brand">
          <span className="header-brand-text">Lumen</span>
        </div>
        <div className="header-actions">
          <button
            onClick={togglePrivacy}
            className={`header-btn ${isAnonymous ? 'header-btn--success' : 'header-btn--warning'}`}
          >
            {isAnonymous ? <EyeOff size={16} /> : <Eye size={16} />} <span>{isAnonymous ? 'Privacy: ON' : 'Privacy: OFF'}</span>
          </button>

          <button
            onClick={handleLogout}
            className="header-btn header-btn--danger"
          >
            <LogOut size={16} /> <span>Logout</span>
          </button>
        </div>
      </header>

      {/* Mobile backdrop overlay */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={closeSidebar} aria-hidden="true" />
      )}

      {/* Content Wrapper - Sidebar + Main side-by-side */}
      <div className="content-wrapper">
        {/* Sidebar Navigation */}
        <aside className={`sidebar${sidebarOpen ? ' sidebar--open' : ''}`}>
          <nav className="sidebar-nav">
            {!isAdmin && (
              <NavLink
                to="/dashboard"
                end
                className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
                onClick={closeSidebar}
              >
                My Dashboard
              </NavLink>
            )}
            {isAdmin && (
              <>
                <NavLink to="/dashboard" end className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`} onClick={closeSidebar}>
                  Overview
                </NavLink>
                <NavLink to="/dashboard/patients" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`} onClick={closeSidebar}>
                  <Users size={18} /> Patients
                </NavLink>
                <NavLink to="/dashboard/reports" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`} onClick={closeSidebar}>
                  <BarChart3 size={18} /> Reports
                </NavLink>
                <NavLink to="/dashboard/voice" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`} onClick={closeSidebar}>
                  <Mic size={18} /> Voice AI
                </NavLink>
              </>
            )}
            <NavLink to="/dashboard/settings" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`} onClick={closeSidebar}>
              <SettingsIcon size={18} /> Settings
            </NavLink>
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={isAdmin ? <Overview /> : <UserDashboard />} />
            {isAdmin && (
              <>
                <Route path="patients" element={<PatientList />} />
                <Route path="patients/:id" element={<PatientDetail />} />
                <Route path="reports" element={<Reports />} />
                <Route path="voice" element={
                  <div className="animate-fade-in">
                    <div className="page-header">
                      <div>
                        <h1 className="page-title">Voice Assessment</h1>
                        <p style={{ color: 'var(--text-secondary)' }}>Run AI voice analysis and log patient scores</p>
                      </div>
                    </div>
                    <div className="glass-panel" style={{ padding: '2rem' }}>
                      <VoiceAssessment isAdmin={true} />
                    </div>
                  </div>
                } />
              </>
            )}
            <Route path="settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
};
