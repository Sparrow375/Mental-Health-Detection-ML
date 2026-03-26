import React, { useState } from 'react';
import { Download, AlertTriangle, Users, TrendingUp } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const Reports: React.FC = () => {
  const [downloading, setDownloading] = useState(false);

  // Dummy aggregate data
  const weeklyAnomalies = [
    { day: 'Mon', anomalies: 2 },
    { day: 'Tue', anomalies: 5 },
    { day: 'Wed', anomalies: 1 },
    { day: 'Thu', anomalies: 8 },
    { day: 'Fri', anomalies: 3 },
    { day: 'Sat', anomalies: 0 },
    { day: 'Sun', anomalies: 4 },
  ];

  const handleExportCSV = () => {
    setDownloading(true);
    setTimeout(() => {
      // Mock CSV generation and download
      const csvContent = "data:text/csv;charset=utf-8,PatientID,Date,Status,AnomalyScore\nPatient_A1B2,2026-03-14,Flagged,0.85\nPatient_C3D4,2026-03-14,Monitoring,0.12";
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "mhealth_system_report.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setDownloading(false);
    }, 1000);
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">System Reports</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Aggregate system analytics and data export</p>
        </div>
        <button 
          onClick={handleExportCSV} 
          disabled={downloading}
          className="btn btn-primary"
        >
          <Download size={18} /> {downloading ? 'Compiling...' : 'Export Full CSV'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <TrendingUp size={18} color="var(--accent-primary)" /> Anomalies Detected (Last 7 Days)
          </h3>
          <div style={{ flex: 1, minHeight: '250px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeklyAnomalies}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="day" stroke="var(--text-muted)" fontSize={12} tickMargin={10} />
                <YAxis stroke="var(--text-muted)" fontSize={12} allowDecimals={false} />
                <Tooltip 
                  cursor={{ fill: 'var(--bg-card-hover)' }}
                  contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                />
                <Bar dataKey="anomalies" fill="var(--accent-secondary)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={18} color="var(--warning)" /> System Alerts
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid var(--danger)', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: 600, color: 'var(--danger)', marginBottom: '0.25rem' }}>Critical System Threshold</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Patient_A1B2 has exceeded the 0.8 anomaly threshold for 3 consecutive days. Direct intervention recommended.</div>
            </div>
            <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', borderLeft: '4px solid var(--accent-primary)', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent-primary)', marginBottom: '0.25rem' }}>Data Volume Report</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>System ingested 124 distinct sync operations from 3 active devices over the last 24 hours.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
