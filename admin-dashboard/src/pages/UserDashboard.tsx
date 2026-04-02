import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Heart, Activity, BellRing, Brain, Smartphone, Footprints, ShieldCheck, MapPin, Map, Users, Compass, Moon, Phone, TrendingUp, TrendingDown, Minus, Wifi, Battery, Clock, Bell, Download, Lock, Database, Cpu, ChevronLeft, ChevronRight, Calendar } from 'lucide-react';

import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
  BarChart, Bar, Cell
} from 'recharts';
import { auth, db } from '../firebase/config';
import { doc, getDoc } from 'firebase/firestore';
import { getHistoricalResults, getAllDailyFeatures, getBaseline } from '../firebase/dataHelper';
import type { MLResult, DailyFeatures, BaselineData } from '../firebase/dataHelper';

export const UserDashboard: React.FC = () => {
  const [patientName, setPatientName] = useState('Patient');
  const [loading, setLoading] = useState(true);
  const [baselineReady, setBaselineReady] = useState(false);

  const [history, setHistory] = useState<MLResult[]>([]);
  const [allDays, setAllDays] = useState<DailyFeatures[]>([]);
  const [selectedDayIndex, setSelectedDayIndex] = useState(0);
  const [baseline, setBaseline] = useState<BaselineData | null>(null);

  const features = allDays.length > 0 ? allDays[selectedDayIndex] : null;

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const currentUser = auth.currentUser;
        if (!currentUser) return;
        
        const uid = currentUser.uid;

        // Fetch User Name
        const docRef = doc(db, 'users', uid);
        const docSnap = await getDoc(docRef);
        if (docSnap.exists()) {
          if (docSnap.data().name) setPatientName(docSnap.data().name);
          if (docSnap.data().baseline_ready === true) setBaselineReady(true);
        }

        // Fetch Real Data
        const histData = await getHistoricalResults(uid, 14);
        setHistory(histData);

        const allFeatures = await getAllDailyFeatures(uid);
        const baselineData = await getBaseline(uid);

        setAllDays(allFeatures);
        setBaseline(baselineData);

      } catch (err) {
        console.error("Failed to fetch user data:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchUserData();
  }, []);

  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem' }}><Activity className="animate-pulse" size={48} color="var(--accent-primary)" /></div>;
  }

  const currentResult = history.length > 0 ? history[history.length - 1] : null;
  const currentScore = currentResult?.anomaly_score || 0;
  const matchMessage = currentResult?.match_message;

  const isCritical = currentScore >= 0.7;
  const isElevated = currentScore >= 0.4 && currentScore < 0.7;

  // Categorized Clinical Biomarkers Data array
  const categorizedFeatures = useMemo(() => [
    {
      category: 'Device & App Usage',
      items: [
        { name: 'Screen Time', value: features?.screenTimeHours, displayValue: features?.screenTimeHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Smartphone, defaultColor: '#38bdf8', key: 'screenTimeHours', invertGood: true },
        { name: 'Unlock Count', value: features?.unlockCount, displayValue: features?.unlockCount?.toFixed(0) || '0', unit: 'times', icon: Lock, defaultColor: '#ef4444', key: 'unlockCount', invertGood: true },
        { name: 'App Launches', value: features?.appLaunchCount, displayValue: features?.appLaunchCount?.toFixed(0) || '0', unit: 'times', icon: Activity, defaultColor: '#f59e0b', key: 'appLaunchCount', invertGood: true },
        { name: 'Total Apps', value: features?.totalAppsCount, displayValue: features?.totalAppsCount?.toFixed(0) || '0', unit: 'apps', icon: Database, defaultColor: '#6366f1', key: 'totalAppsCount', invertGood: false },
        { name: 'App Uninstalls', value: features?.appUninstallsToday, displayValue: features?.appUninstallsToday?.toFixed(0) || '0', unit: 'apps', icon: Minus, defaultColor: '#ef4444', key: 'appUninstallsToday', invertGood: true },
        { name: 'UPI Txns', value: features?.upiTransactionsToday, displayValue: features?.upiTransactionsToday?.toFixed(0) || '0', unit: 'txns', icon: Activity, defaultColor: '#10b981', key: 'upiTransactionsToday', invertGood: false },
      ]
    },
    {
      category: 'Social & Communication',
      items: [
        { name: 'Notifications', value: features?.notificationsToday, displayValue: features?.notificationsToday?.toFixed(0) || '0', unit: 'alerts', icon: Bell, defaultColor: '#ec4899', key: 'notificationsToday', invertGood: true },
        { name: 'Social App Ratio', value: features?.socialAppRatio, displayValue: features?.socialAppRatio ? (features.socialAppRatio * 100).toFixed(0) : '0', unit: '%', icon: Users, defaultColor: '#ec4899', key: 'socialAppRatio', invertGood: false },
        { name: 'Calls per Day', value: features?.callsPerDay, displayValue: features?.callsPerDay?.toFixed(0) || '0', unit: 'calls', icon: Phone, defaultColor: '#facc15', key: 'callsPerDay', invertGood: false },
        { name: 'Call Duration', value: features?.callDurationMinutes, displayValue: features?.callDurationMinutes?.toFixed(0) || '0', unit: 'mins', icon: Phone, defaultColor: '#facc15', key: 'callDurationMinutes', invertGood: false },
        { name: 'Unique Contacts', value: features?.uniqueContacts, displayValue: features?.uniqueContacts?.toFixed(0) || '0', unit: 'ppl', icon: Users, defaultColor: '#10b981', key: 'uniqueContacts', invertGood: false },
        { name: 'Conv. Frequency', value: features?.conversationFrequency, displayValue: features?.conversationFrequency?.toFixed(2) || '0.0', unit: 'freq', icon: Activity, defaultColor: '#8b5cf6', key: 'conversationFrequency', invertGood: false }
      ]
    },
    {
      category: 'Mobility & Location',
      items: [
        { name: 'Displacement', value: features?.dailyDisplacementKm, displayValue: features?.dailyDisplacementKm?.toFixed(1) || '0.0', unit: 'km', icon: Map, defaultColor: '#a855f7', key: 'dailyDisplacementKm', invertGood: false },
        { name: 'Places Visited', value: features?.placesVisited, displayValue: features?.placesVisited?.toFixed(0) || '0', unit: 'locations', icon: MapPin, defaultColor: 'var(--warning)', key: 'placesVisited', invertGood: false },
        { name: 'Location Entropy', value: features?.locationEntropy, displayValue: features?.locationEntropy?.toFixed(2) || '0.00', unit: 'score', icon: Compass, defaultColor: '#14b8a6', key: 'locationEntropy', invertGood: false },
        { name: 'Home Time Ratio', value: features?.homeTimeRatio, displayValue: features?.homeTimeRatio ? (features.homeTimeRatio * 100).toFixed(0) : '0', unit: '%', icon: Clock, defaultColor: '#38bdf8', key: 'homeTimeRatio', invertGood: true },
      ]
    },
    {
      category: 'Sleep & Activity',
      items: [
        { name: 'Step Count', value: features?.dailySteps, displayValue: features?.dailySteps?.toFixed(0) || '0', unit: 'steps', icon: Footprints, defaultColor: 'var(--success)', key: 'dailySteps', invertGood: false },
        { name: 'Sleep Duration', value: features?.sleepDurationHours, displayValue: features?.sleepDurationHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Moon, defaultColor: '#6366f1', key: 'sleepDurationHours', invertGood: false },
        { name: 'Wake Time', value: features?.wakeTimeHour, displayValue: features?.wakeTimeHour?.toFixed(1) || '0.0', unit: 'hr', icon: Clock, defaultColor: '#f59e0b', key: 'wakeTimeHour', invertGood: false },
        { name: 'Sleep Time', value: features?.sleepTimeHour, displayValue: features?.sleepTimeHour?.toFixed(1) || '0.0', unit: 'hr', icon: Moon, defaultColor: '#6366f1', key: 'sleepTimeHour', invertGood: false },
      ]
    },
    {
      category: 'System & Resources',
      items: [
        { name: 'Dark Duration', value: features?.darkDurationHours, displayValue: features?.darkDurationHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Moon, defaultColor: '#64748b', key: 'darkDurationHours', invertGood: false },
        { name: 'Charge Duration', value: features?.chargeDurationHours, displayValue: features?.chargeDurationHours?.toFixed(1) || '0.0', unit: 'hrs', icon: Battery, defaultColor: '#10b981', key: 'chargeDurationHours', invertGood: false },
        { name: 'Memory Usage', value: features?.memoryUsagePercent, displayValue: features?.memoryUsagePercent?.toFixed(0) || '0', unit: '%', icon: Cpu, defaultColor: '#ef4444', key: 'memoryUsagePercent', invertGood: true },
        { name: 'Storage Used', value: features?.storageUsedGB, displayValue: features?.storageUsedGB?.toFixed(1) || '0.0', unit: 'GB', icon: Database, defaultColor: '#6366f1', key: 'storageUsedGB', invertGood: true },
        { name: 'Wi-Fi Data', value: features?.networkWifiMB, displayValue: features?.networkWifiMB?.toFixed(0) || '0', unit: 'MB', icon: Wifi, defaultColor: '#38bdf8', key: 'networkWifiMB', invertGood: true },
        { name: 'Mobile Data', value: features?.networkMobileMB, displayValue: features?.networkMobileMB?.toFixed(0) || '0', unit: 'MB', icon: Activity, defaultColor: '#8b5cf6', key: 'networkMobileMB', invertGood: true },
        { name: 'Downloads', value: features?.downloadsToday, displayValue: features?.downloadsToday?.toFixed(0) || '0', unit: 'files', icon: Download, defaultColor: '#f59e0b', key: 'downloadsToday', invertGood: true },
      ]
    }
  ], [features]);

  const getClinicalStatus = useCallback((featureName: string, currentVal: number | undefined, invert: boolean) => {
    if (currentVal === undefined) return { color: 'var(--text-muted)', status: 'No Data', icon: null };
    if (!baselineReady || !baseline || !baseline[featureName] || typeof baseline[featureName].mean !== 'number') return { color: 'var(--text-secondary)', status: 'Building...', icon: <Minus size={14} /> };
    
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
  }, [baselineReady, baseline]);

  // Standardized Z-Score calculation for bar chart
  const getZScore = useCallback((featureName: string): number => {
    if (!baselineReady || !features || !baseline || !baseline[featureName] || typeof baseline[featureName].mean !== 'number') return 0; 
    const currentVal = (features as any)[featureName] || 0;
    const baseMean = baseline[featureName].mean;
    const baseStd = baseline[featureName].std || (baseMean * 0.1) || 1;
    let z = (currentVal - baseMean) / baseStd;
    return Math.min(Math.max(z, -4), 4); // Cap at -4 and +4 for chart readability
  }, [baselineReady, features, baseline]);

  const featureDeviations = useMemo(() => [
    { name: 'Screen Time', zScore: getZScore('screenTimeHours') },
    { name: 'Sociability', zScore: getZScore('socialAppRatio') },
    { name: 'Mobility', zScore: getZScore('placesVisited') },
    { name: 'Loc. Entropy', zScore: getZScore('locationEntropy') },
    { name: 'Sleep Dur.', zScore: getZScore('sleepDurationHours') },
    { name: 'App Launches', zScore: getZScore('appLaunchCount') },
    { name: 'Comm. Freq.', zScore: getZScore('conversationFrequency') },
    { name: 'Steps', zScore: getZScore('dailySteps') },
    { name: 'Home Time', zScore: getZScore('homeTimeRatio') },
    { name: 'Mem. Usage', zScore: getZScore('memoryUsagePercent') }
  ].filter(f => f.zScore !== 0).sort((a, b) => b.zScore - a.zScore), [getZScore]);

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      
      {/* Greeting Header */}
      <div style={{ marginBottom: '2.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Welcome back.
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            Here is the real-time behavioral overview for <strong>{patientName}</strong>.
          </p>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: baseline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)', color: baseline ? 'var(--success)' : 'var(--warning)', padding: '0.75rem 1.25rem', borderRadius: '2rem', fontWeight: 600 }}>
          <ShieldCheck size={20} />
          {baseline ? 'Baseline Established' : 'Building Baseline...'}
        </div>
      </div>

      {/* Comprehensive Clinical Metrics Grid */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem', marginLeft: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={20} color="var(--accent-primary)" /> Clinical Biomarkers (24h Window)
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', marginBottom: '2.5rem' }}>
        {categorizedFeatures.map((categoryGroup, index) => (
          <div key={index}>
            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', color: 'var(--text-primary)', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
              {categoryGroup.category}
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              {categoryGroup.items.map((feat) => {
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
          </div>
        ))}
      </div>

      {/* Main Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* Longitudinal History */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Activity size={18} color="var(--accent-primary)" /> Wellness Stability (Last 14 Days)
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem' }}>
             Tracking behavioral consistency via the ML anomaly engine. Lower is better.
          </p>

          <div style={{ flex: 1, minHeight: '300px' }}>
            {history.length === 0 ? (
               <div style={{ height: '100%', display: 'flex', alignItems:'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Synching baseline data...</div>
            ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 10, right: 0, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--danger)" stopOpacity={0.6}/>
                    <stop offset="60%" stopColor="var(--warning)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--success)" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={12} tickMargin={10} />
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} tickFormatter={(v: any) => typeof v === 'number' ? v.toFixed(1) : v} />
                <Tooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg)' }}
                  itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                  labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}
                />
                
                <ReferenceLine y={0.7} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'High Variance', fill: 'var(--danger)', fontSize: 11, fontWeight: 600 }} />
                
                <Area type="monotone" dataKey="anomaly_score" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" activeDot={{ r: 6, fill: 'var(--accent-primary)', stroke: '#fff', strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
            )}
          </div>
        </div>
        
        {/* Diagnostic Bar Chart (Deviations) */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Brain size={18} color="var(--accent-primary)" /> Behavioral Deviation (Z-Scores)
          </h3>
          <div style={{ flex: 1, minHeight: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
            {!baselineReady || !baseline || !features ? (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center' }}>{!baselineReady ? 'Baseline period not yet complete' : 'No feature data available'}</div>
            ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureDeviations} margin={{ top: 20, right: 10, left: 10, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={true} vertical={false} />
                <XAxis dataKey="name" type="category" stroke="var(--text-muted)" fontSize={10} angle={-45} textAnchor="end" interval={0} tickMargin={5} />
                <YAxis type="number" stroke="var(--text-muted)" fontSize={12} tickCount={5} domain={[-4, 4]} tickFormatter={(v: any) => `${v > 0 ? '+' : ''}${v}σ`} />
                <Tooltip 
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                  itemStyle={{ fontWeight: 600 }}
                  formatter={(val: any) => [`${val > 0 ? '+' : ''}${Number(val).toFixed(2)}σ`, 'Z-Score']}
                />
                
                {/* 0 Baseline Reference Line */}
                <ReferenceLine y={0} stroke="var(--text-primary)" strokeWidth={2} />
                <ReferenceLine y={2} stroke="var(--warning)" strokeDasharray="3 3" />
                <ReferenceLine y={-2} stroke="var(--warning)" strokeDasharray="3 3" />
                
                <Bar dataKey="zScore" radius={[4, 4, 0, 0]}>
                  {featureDeviations.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.zScore > 0 ? 'var(--warning)' : 'var(--success)'} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem', marginLeft: '0.5rem' }}>Dynamic Insights</h3>
      
      {/* ML Engine Message */}
      {matchMessage ? (
         <div style={{ 
            background: 'var(--bg-card)', 
            border: `1px solid var(--border)`, 
            borderLeft: `4px solid var(--accent-primary)`,
            borderRadius: 'var(--radius-md)',
            padding: '1.5rem',
            display: 'flex',
            gap: '1rem',
            alignItems: 'flex-start',
            marginBottom: '1rem'
          }}>
            <div style={{ background: 'rgba(56, 189, 248, 0.1)', padding: '0.75rem', borderRadius: '50%', flexShrink: 0 }}>
              <Heart size={20} color="var(--accent-primary)" />
            </div>
            <div>
              <h4 style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.25rem' }}>System Health Update</h4>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                 {matchMessage}
              </p>
            </div>
          </div>
      ) : (
         <div style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>Gathering insights...</div>
      )}

      <div style={{ marginTop: '3rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <BellRing size={24} color="var(--text-muted)" style={{ marginBottom: '1rem' }} />
        <p style={{ color: 'var(--text-muted)', maxWidth: '400px' }}>
           Urgent alerts and clinical anomalies will be displayed here securely.
        </p>
      </div>
    </div>
  );
};
