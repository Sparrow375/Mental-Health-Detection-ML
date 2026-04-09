import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Heart, Activity, BellRing, Brain, Smartphone, Footprints, ShieldCheck, MapPin, Map, Users, Compass, Moon, Phone, TrendingUp, TrendingDown, Minus, Wifi, Battery, Clock, Bell, Download, Lock, Database, Cpu } from 'lucide-react';

import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts';
import { auth, db } from '../firebase/config';
import { doc, getDoc } from 'firebase/firestore';
import { getHistoricalResults, getAllDailyFeatures, getBaseline } from '../firebase/dataHelper';
import type { MLResult, DailyFeatures, BaselineData } from '../firebase/dataHelper';
import { BaselineSlopeChart } from '../components/BaselineLineGraph';

export const UserDashboard: React.FC = () => {
  const [patientName, setPatientName] = useState('Patient');
  const [baselineReady, setBaselineReady] = useState(false);
  const [loading, setLoading] = useState(true);

  const [history, setHistory] = useState<MLResult[]>([]);
  const [allDays, setAllDays] = useState<DailyFeatures[]>([]);
  const [baseline, setBaseline] = useState<BaselineData | null>(null);

  const selectedDayIndex = 0; // For user dashboard, always show latest day
  const features = allDays.length > 0 ? allDays[selectedDayIndex] : null;

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const currentUser = auth.currentUser;
        if (!currentUser) {
          setLoading(false);
          return;
        }

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

  const currentResult = history.length > 0 ? history[history.length - 1] : null;
  const matchMessage = currentResult?.match_message;

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


  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <Activity className="animate-spin" size={48} color="var(--accent-primary)" />
      </div>
    );
  }

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

      {/* Comprehensive Clinical Biomarkers Table */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem', marginLeft: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={20} color="var(--accent-primary)" /> Clinical Biomarkers (24h Window)
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', marginBottom: '2.5rem' }}>
        {categorizedFeatures.map((categoryGroup, index) => (
          <div key={index} className="glass-panel" style={{ overflow: 'hidden' }}>
            <h4 style={{ padding: '1rem 1.5rem', margin: 0, borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.1)', color: 'var(--text-primary)', fontWeight: 600 }}>
              {categoryGroup.category}
            </h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '0.75rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase' }}>Biomarker / Feature</th>
                  <th style={{ padding: '0.75rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase' }}>Current Score</th>
                  <th style={{ padding: '0.75rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase' }}>Baseline Mean</th>
                  <th style={{ padding: '0.75rem 1.5rem', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase' }}>Variance (Z-Score)</th>
                  <th style={{ padding: '0.75rem 1.5rem', textAlign: 'right', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase' }}>Clinical Status</th>
                </tr>
              </thead>
              <tbody>
                {categoryGroup.items.map((feat) => {
                  const statusObj = getClinicalStatus(feat.key, feat.value, feat.invertGood);
                  const baselineMean = baseline && baseline[feat.key] && typeof baseline[feat.key].mean === 'number' ? baseline[feat.key].mean : null;
                  const baselineStd = baseline && baseline[feat.key] && typeof baseline[feat.key].std === 'number' ? baseline[feat.key].std : (baselineMean !== null ? Math.max(baselineMean * 0.1, 1) : null);

                  let varianceDisplay = 'N/A';
                  if (feat.value !== undefined && baselineMean !== null && baselineStd !== null && baselineStd > 0) {
                      const diff = feat.value - baselineMean;
                      const z = (diff / baselineStd).toFixed(2);
                      varianceDisplay = `${diff > 0 ? '+' : ''}${z}σ`;
                  }

                  const Icon = feat.icon;
                  return (
                    <tr
                      key={feat.key}
                      style={{ borderBottom: '1px solid var(--border)', transition: 'background var(--transition-fast)' }}
                      onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-card-hover)'}
                      onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      <td style={{ padding: '1rem 1.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                           <div style={{ background: `${feat.defaultColor}15`, padding: '0.4rem', borderRadius: '0.5rem', color: feat.defaultColor, display: 'flex' }}>
                             <Icon size={16} />
                           </div>
                           <span style={{ fontWeight: 500 }}>{feat.name}</span>
                        </div>
                      </td>
                      <td style={{ padding: '1rem 1.5rem', fontWeight: 600, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                        {feat.displayValue} <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-muted)' }}>{feat.unit}</span>
                      </td>
                      <td style={{ padding: '1rem 1.5rem', color: 'var(--text-secondary)' }}>
                        {baselineMean !== null ? baselineMean.toFixed(1) : 'Est...'} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{baselineMean !== null ? feat.unit : ''}</span>
                      </td>
                      <td style={{ padding: '1rem 1.5rem', fontFamily: 'monospace', color: varianceDisplay.includes('+') ? (feat.invertGood ? 'var(--danger)' : 'var(--warning)') : (varianceDisplay.includes('-') ? (feat.invertGood ? 'var(--success)' : 'var(--danger)') : 'var(--text-muted)') }}>
                        {varianceDisplay}
                      </td>
                      <td style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', background: `${statusObj.color}15`, color: statusObj.color, fontSize: '0.875rem', fontWeight: 600 }}>
                          {statusObj.icon} {statusObj.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} tickFormatter={(v: unknown) => typeof v === 'number' ? v.toFixed(1) : String(v)} />
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

        {/* Baseline Comparison - Line Graph Visualization */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Brain size={18} color="var(--accent-primary)" /> Baseline Comparison
          </h3>
          {!baselineReady || !baseline || !features ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
              {!baselineReady ? 'Baseline period not yet complete' : 'No feature data available'}
            </div>
          ) : (
            <BaselineSlopeChart
              metrics={[
                {
                  label: 'Screen Time',
                  current: features.screenTimeHours || 0,
                  baseline: baseline.screenTimeHours?.mean || 0,
                  unit: 'hrs',
                  invertGood: true
                },
                {
                  label: 'Sleep Duration',
                  current: features.sleepDurationHours || 0,
                  baseline: baseline.sleepDurationHours?.mean || 0,
                  unit: 'hrs'
                },
                {
                  label: 'Social Ratio',
                  current: (features.socialAppRatio || 0) * 100,
                  baseline: (baseline.socialAppRatio?.mean || 0) * 100,
                  unit: '%'
                },
                {
                  label: 'Daily Steps',
                  current: features.dailySteps || 0,
                  baseline: baseline.dailySteps?.mean || 0,
                  unit: 'steps'
                },
                {
                  label: 'App Launches',
                  current: features.appLaunchCount || 0,
                  baseline: baseline.appLaunchCount?.mean || 0,
                  unit: 'times',
                  invertGood: true
                }
              ]}
            />
          )}
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
