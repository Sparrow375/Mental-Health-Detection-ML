import React, { useState, useEffect } from 'react';
import { Heart, Activity, BellRing, Moon, MessageCircle, Footprints, ShieldCheck } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { auth, db } from '../firebase/config';
import { doc, getDoc } from 'firebase/firestore';

export const UserDashboard: React.FC = () => {
  const [patientName, setPatientName] = useState('Patient');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatientName = async () => {
      try {
        const currentUser = auth.currentUser;
        if (currentUser) {
          const docRef = doc(db, 'users', currentUser.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists() && docSnap.data().name) {
            setPatientName(docSnap.data().name);
          }
        }
      } catch (err) {
        console.error("Failed to fetch user:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchPatientName();
  }, []);

  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem' }}><Activity className="animate-pulse" size={48} color="var(--accent-primary)" /></div>;
  }

  // Mock data for the Caregiver MVP
  const childData = {
    name: patientName,
    overallWellness: 'Stable', // Compassionate language
    recentActivity: [
      { date: 'Mon', wellnessScore: 88 },
      { date: 'Tue', wellnessScore: 85 },
      { date: 'Wed', wellnessScore: 90 },
      { date: 'Thu', wellnessScore: 82 },
      { date: 'Fri', wellnessScore: 78 },
      { date: 'Sat', wellnessScore: 86 },
      { date: 'Sun', wellnessScore: 89 },
    ],
    insights: [
      {
        icon: <Moon size={20} color="var(--warning)" />,
        title: 'Mild Sleep Disruption',
        message: `We noticed ${patientName} was slightly restless last night. It might be a good time to check in and see if they are feeling stressed.`,
        color: 'var(--warning)',
        bgColor: 'rgba(245, 158, 11, 0.05)',
        borderColor: 'rgba(245, 158, 11, 0.2)',
      },
      {
        icon: <Footprints size={20} color="var(--success)" />,
        title: 'Great Physical Activity',
        message: `${patientName} has been keeping up with their physical activities beautifully over the past 3 days!`,
        color: 'var(--success)',
        bgColor: 'rgba(16, 185, 129, 0.05)',
        borderColor: 'rgba(16, 185, 129, 0.2)',
      },
      {
        icon: <MessageCircle size={20} color="var(--accent-primary)" />,
        title: 'Social Engagement is Consistent',
        message: 'Social and communication levels are remaining consistent with his personal baseline.',
        color: 'var(--accent-primary)',
        bgColor: 'rgba(59, 130, 246, 0.05)',
        borderColor: 'rgba(59, 130, 246, 0.2)',
      }
    ]
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '900px', margin: '0 auto' }}>
      
      {/* Compassionate Greeting */}
      <div style={{ marginBottom: '2.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Welcome back.
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            Here is your behavioral wellness update for <strong>{childData.name}</strong>.
          </p>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)', padding: '0.75rem 1.25rem', borderRadius: '2rem', fontWeight: 600 }}>
          <ShieldCheck size={20} />
          Monitoring Active
        </div>
      </div>

      {/* Primary Wellness Card */}
      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem', background: 'var(--bg-card)', borderTop: '4px solid var(--accent-primary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Heart size={28} color="var(--accent-primary)" fill="var(--accent-primary)" />
            <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Overall Wellness: {childData.overallWellness}</h2>
          </div>
          <p style={{ color: 'var(--text-muted)' }}>Updated 2 hours ago</p>
        </div>
        
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: 1.6, marginBottom: '2rem' }}>
          {childData.name} has had a very balanced week. There are no major shifts in their behavioral patterns. Based on the passive data collected from their device, things are looking good.
        </p>

        {/* 7-Day Wellness Trend - Soft colors instead of clinical red/yellow/green zones */}
        <div style={{ height: '200px', marginTop: '1rem' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={childData.recentActivity} margin={{ top: 10, right: 0, bottom: 0, left: -20 }}>
              <defs>
                <linearGradient id="colorWellness" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} hide={true} />
              <Tooltip 
                contentStyle={{ background: 'var(--bg-primary)', border: 'none', borderRadius: '1rem', boxShadow: 'var(--shadow-md)' }}
                itemStyle={{ color: 'var(--accent-primary)', fontWeight: 600 }}
                labelStyle={{ display: 'none' }}
                formatter={(value: any) => [`${value}% Wellness`, '']}
              />
              <Area type="monotone" dataKey="wellnessScore" stroke="var(--accent-primary)" strokeWidth={4} fillOpacity={1} fill="url(#colorWellness)" activeDot={{ r: 6, stroke: '#fff', strokeWidth: 3 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem', marginLeft: '0.5rem' }}>Weekly Check-in Suggestions</h3>
      
      {/* Compassionate Insights Engine */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {childData.insights.map((insight, idx) => (
          <div key={idx} style={{ 
            background: insight.bgColor, 
            border: `1px solid ${insight.borderColor}`, 
            borderLeft: `4px solid ${insight.color}`,
            borderRadius: 'var(--radius-md)',
            padding: '1.5rem',
            display: 'flex',
            gap: '1rem',
            alignItems: 'flex-start'
          }}>
            <div style={{ background: '#fff', padding: '0.75rem', borderRadius: '50%', border: `1px solid ${insight.borderColor}`, flexShrink: 0 }}>
              {insight.icon}
            </div>
            <div>
              <h4 style={{ color: insight.color, fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.25rem' }}>{insight.title}</h4>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {insight.message}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '3rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <BellRing size={24} color="var(--text-muted)" style={{ marginBottom: '1rem' }} />
        <p style={{ color: 'var(--text-muted)', maxWidth: '400px' }}>
          You are currently receiving push notifications for urgent alerts. You can configure your notification preferences in the Settings tab.
        </p>
      </div>
    </div>
  );
};
