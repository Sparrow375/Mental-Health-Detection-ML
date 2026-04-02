import React, { useState, useEffect } from 'react';
import { usePrivacy } from '../context/PrivacyContext';
import { Settings as SettingsIcon, Shield, Palette, Database, Bell, Save } from 'lucide-react';

export const Settings: React.FC = () => {
  const { isAnonymous, togglePrivacy } = usePrivacy();
  const [theme, setTheme] = useState<'system'|'dark'>('dark');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Load local preferences
    const savedTheme = localStorage.getItem('themePref') || 'dark';
    setTheme(savedTheme as any);
  }, []);

  const handleSave = () => {
    localStorage.setItem('themePref', theme);
    // Apply changes
    setSaved(true);
    setTimeout(() => {
        setSaved(false);
    }, 1500);
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '800px' }}>
      <div className="page-header" style={{ marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <SettingsIcon size={28} /> Preferences
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>Manage your dashboard experience and privacy controls.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '1.5rem' }}>
        
        {/* PRIVACY & SECURITY */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
            <Shield size={20} color="var(--accent-primary)" /> Privacy & Security
          </h2>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>PHI Anonymization Mode</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Mask all Protected Health Information (Names, Emails) from the view.</div>
            </div>
            <button
              onClick={togglePrivacy}
              style={{
                background: isAnonymous ? 'var(--success)' : 'var(--bg-card-hover)',
                color: isAnonymous ? '#fff' : 'var(--text-primary)',
                border: '1px solid var(--border)',
                padding: '0.5rem 1.5rem',
                borderRadius: '2rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {isAnonymous ? 'Enabled' : 'Disabled'}
            </button>
          </div>
        </div>

        {/* APPEARANCE */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
            <Palette size={20} color="var(--accent-primary)" /> Appearance
          </h2>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Color Theme</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Choose a visual mode for the dashboard.</div>
            </div>
            <select 
              value={theme}
              onChange={(e) => setTheme(e.target.value as any)}
              style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: 'var(--bg-card-hover)', border: '1px solid var(--border)', color: 'var(--text-primary)', cursor: 'pointer' }}
            >
              <option value="system">System Default</option>
              <option value="dark">Clinical Dark Mode</option>
            </select>
          </div>
        </div>



      </div>

      <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
        <button 
          onClick={handleSave}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: saved ? 'var(--success)' : 'var(--accent-primary)',
            color: '#fff',
            border: 'none',
            padding: '0.75rem 1.5rem',
            borderRadius: 'var(--radius-md)',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          {saved ? <Shield size={18} /> : <Save size={18} />}
          {saved ? 'Preferences Saved' : 'Save Changes'}
        </button>
      </div>

    </div>
  );
};
