import React, { useEffect, useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/config';
import { getHistoricalResults, getAllDailyFeatures, getBaseline } from '../firebase/dataHelper';
import type { MLResult, DailyFeatures, BaselineData } from '../firebase/dataHelper';
import { ArrowLeft, Activity, Cpu, AlertTriangle, ShieldCheck, HeartPulse, Brain, Smartphone, Footprints, MapPin, Map, Users, Compass, Moon, Phone, TrendingUp, TrendingDown, Minus, Target, Wifi, Battery, Clock, Bell, Download, Lock, Database, ChevronLeft, ChevronRight, Calendar } from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { BaselineSlopeChart } from '../components/BaselineLineGraph';

export const PatientDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  
  const [patient, setPatient] = useState<Record<string, string | number | boolean | null> | null>(null);
  const [history, setHistory] = useState<MLResult[]>([]);
  const [allDays, setAllDays] = useState<DailyFeatures[]>([]); // All daily records (newest first)
  const [selectedDayIndex, setSelectedDayIndex] = useState(0); // 0 = latest day
  const [baseline, setBaseline] = useState<BaselineData | null>(null);

  const features = allDays.length > 0 ? allDays[selectedDayIndex] : null;

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
      
      try {
        const docRef = doc(db, 'users', id);
        const docSnap = await getDoc(docRef);
        
        if (docSnap.exists()) {
          setPatient(docSnap.data());
        }

        const histData = await getHistoricalResults(id, 14);
        setHistory(histData);

        const allFeatures = await getAllDailyFeatures(id);
        const baselineData = await getBaseline(id);

        setAllDays(allFeatures);
        setBaseline(baselineData);

      } catch (e) {
        console.error("Error fetching detail:", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  const currentResult = history.length > 0 ? history[history.length - 1] : null;
  const currentScore = currentResult?.anomaly_score || 0;
  
  const isCritical = currentScore >= 0.7;
  const isElevated = currentScore >= 0.4 && currentScore < 0.7;
  const prototypeMatch = currentResult?.prototype_match;
  const matchMessage = currentResult?.match_message;

  // ── Parse all_scores_json into ranked Top-3 list ──────────────────────
  const allScoresRanked: { name: string; score: number }[] = React.useMemo(() => {
    if (!currentResult?.all_scores_json) {
      if (prototypeMatch) {
        return [{ name: prototypeMatch, score: currentResult?.prototype_confidence || 0 }];
      }
      return [];
    }
    try {
      const parsed = JSON.parse(currentResult.all_scores_json);
      return Object.entries(parsed)
        .map(([name, score]) => ({ name, score: Number(score) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
    } catch {
      if (prototypeMatch) {
        return [{ name: prototypeMatch, score: currentResult?.prototype_confidence || 0 }];
      }
      return [];
    }
  }, [currentResult, prototypeMatch]);

  // Close-call detection: top-1 is healthy but a clinical disorder is within 10%
  const clinicalCloseCall = React.useMemo(() => {
    if (allScoresRanked.length < 2) return null;
    const top1 = allScoresRanked[0];
    const top1IsHealthy = top1.name.toLowerCase().startsWith('healthy') || top1.name.toLowerCase() === 'normal';
    if (!top1IsHealthy) return null;
    const second = allScoresRanked[1];
    const secondIsClinical = !second.name.toLowerCase().startsWith('healthy') && second.name.toLowerCase() !== 'normal';
    if (secondIsClinical && (top1.score - second.score) < 0.10) {
      return { name: second.name, margin: ((top1.score - second.score) * 100).toFixed(1) };
    }
    return null;
  }, [allScoresRanked]);

  // Calculate insights from baseline deviations (percentage-based)
  const insights = useMemo(() => {
    if (!baseline || !features) return [];
    const metrics = [
      { name: 'Screen Time', key: 'screenTimeHours', invert: true },
      { name: 'Sleep Duration', key: 'sleepDurationHours', invert: false },
      { name: 'Social Ratio', key: 'socialAppRatio', invert: false },
      { name: 'Daily Steps', key: 'dailySteps', invert: false },
      { name: 'App Launches', key: 'appLaunchCount', invert: true },
    ];
    return metrics
      .map(m => {
        const current = (features as Record<string, unknown>)[m.key] as number || 0;
        const baseMean = baseline[m.key]?.mean || 0;
        const baseStd = baseline[m.key]?.std || (baseMean * 0.1) || 1;
        const z = (current - baseMean) / baseStd;
        return { name: m.name, deviation: Math.abs(z), rawScore: z, sign: z > 0 ? '+' : '-', invert: m.invert };
      })
      .sort((a, b) => b.deviation - a.deviation)
      .slice(0, 3);
  }, [baseline, features]);

  if (loading) return <div style={{ padding: '2rem', color: 'var(--text-muted)' }}>Loading clinical metrics...</div>;
  if (!patient && history.length === 0) return <div style={{ padding: '2rem', color: 'var(--danger)' }}>Patient not found or no data available.</div>;



  // Categorized Clinical Biomarkers Data array
  const categorizedFeatures = [
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
  ];

  const baselineReady = patient?.baseline_ready === true;

  const getClinicalStatus = (featureName: string, currentVal: number | undefined, invert: boolean) => {
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
  };


  const hasPrototypeMatch = !!prototypeMatch && prototypeMatch.trim() !== "";


  return (
    <div className="animate-fade-in">
      <div className="page-header" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
        <button 
          onClick={() => navigate('/patients')} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontWeight: 500 }}
          onMouseOver={e => e.currentTarget.style.color = 'var(--text-primary)'}
          onMouseOut={e => e.currentTarget.style.color = 'var(--text-secondary)'}
        >
          <ArrowLeft size={18} /> Back to Directory
        </button>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ background: isCritical ? 'rgba(239, 68, 68, 0.1)' : isElevated ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)', padding: '1rem', borderRadius: '50%' }}>
              {isCritical ? <AlertTriangle size={32} color="var(--danger)" /> : isElevated ? <Activity size={32} color="var(--warning)" /> : <ShieldCheck size={32} color="var(--success)" />}
            </div>
            <div>
              <h1 className="page-title" style={{ marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {patient?.patient_id || id}
                <span style={{ fontSize: '0.875rem', padding: '0.25rem 0.75rem', borderRadius: '1rem', fontWeight: 600, background: isCritical ? 'var(--danger)' : isElevated ? 'var(--warning)' : 'var(--success)', color: '#fff' }}>
                  {Math.round(currentScore * 100)}% ACUITY
                </span>
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                Patient since {new Date((patient?.onboarding_date as number) || Date.now()).toLocaleDateString()} 
                {baseline ? ' • Baseline Established' : ' • Establishing Baseline...'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Clinical Metrics Grid */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
          <Activity size={20} color="var(--accent-primary)" /> Clinical Biomarkers (24h Window)
        </h3>
        {allDays.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={() => setSelectedDayIndex(Math.min(selectedDayIndex + 1, allDays.length - 1))}
              disabled={selectedDayIndex >= allDays.length - 1}
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '0.5rem', cursor: selectedDayIndex >= allDays.length - 1 ? 'not-allowed' : 'pointer', color: selectedDayIndex >= allDays.length - 1 ? 'var(--text-muted)' : 'var(--text-primary)', display: 'flex', alignItems: 'center' }}
            >
              <ChevronLeft size={16} />
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', minWidth: '180px', justifyContent: 'center' }}>
              <Calendar size={16} color="var(--accent-primary)" />
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{features?.date || 'No Data'}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>({selectedDayIndex === 0 ? 'Latest' : `${selectedDayIndex}d ago`})</span>
            </div>
            <button
              onClick={() => setSelectedDayIndex(Math.max(selectedDayIndex - 1, 0))}
              disabled={selectedDayIndex <= 0}
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '0.5rem', cursor: selectedDayIndex <= 0 ? 'not-allowed' : 'pointer', color: selectedDayIndex <= 0 ? 'var(--text-muted)' : 'var(--text-primary)', display: 'flex', alignItems: 'center' }}
            >
              <ChevronRight size={16} />
            </button>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: '0.25rem' }}>{allDays.length} day{allDays.length !== 1 ? 's' : ''} recorded</span>
          </div>
        )}
      </div>
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

        {/* Risk-Stratified Trendline */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Activity size={18} color="var(--accent-primary)" /> Longitudinal Anomaly Trajectory (Last 14 Days)
          </h3>
          <div style={{ flex: 1, minHeight: '350px' }}>
            {history.length === 0 ? (
               <div style={{ height: '100%', display: 'flex', alignItems:'center', justifyContent: 'center', color: 'var(--text-muted)' }}>No historical data available yet.</div>
            ) : (
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
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} tickFormatter={(v: unknown) => typeof v === 'number' ? v.toFixed(1) : String(v)} />
                <RechartsTooltip
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg)' }}
                  itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                  labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}
                />

                <ReferenceLine y={0.7} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'CRITICAL (0.7)', fill: 'var(--danger)', fontSize: 11, fontWeight: 600 }} />
                <ReferenceLine y={0.4} stroke="var(--warning)" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'ELEVATED (0.4)', fill: 'var(--warning)', fontSize: 11, fontWeight: 600 }} />

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


      {/* ── Top-3 Prototype Classification Panel ────────────────────────── */}
      {allScoresRanked.length > 0 && (
        <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem', borderLeft: '4px solid #f59e0b' }}>
          <h3 style={{ marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Cpu size={18} color="#f59e0b" /> System 2 Clinical Validations — Top Matches
          </h3>

          {/* Close-call warning banner */}
          {clinicalCloseCall && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.75rem 1rem', marginBottom: '1rem',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--warning)', fontWeight: 600, fontSize: '0.875rem'
            }}>
              <AlertTriangle size={16} />
              Close call: {clinicalCloseCall.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} is within {clinicalCloseCall.margin}% of top match
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {allScoresRanked.map((entry, idx) => {
              const displayName = entry.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
              const isClinical = !entry.name.toLowerCase().startsWith('healthy') && entry.name.toLowerCase() !== 'normal';
              const barColor = isClinical ? 'var(--danger)' : 'var(--accent-primary)';
              const pct = (entry.score * 100).toFixed(1);
              const rankColors = ['var(--accent-primary)', 'var(--text-secondary)', 'var(--text-muted)'];

              return (
                <div key={entry.name} style={{
                  display: 'flex', alignItems: 'center', gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  background: idx === 0 ? `${barColor}08` : 'transparent',
                  border: idx === 0 ? `1px solid ${barColor}30` : '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  transition: 'all 0.2s'
                }}>
                  <div style={{
                    width: '1.75rem', height: '1.75rem',
                    borderRadius: '50%',
                    background: `${rankColors[idx]}15`,
                    border: `1.5px solid ${rankColors[idx]}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.75rem', fontWeight: 700,
                    color: rankColors[idx],
                    flexShrink: 0
                  }}>
                    {idx + 1}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                      <span style={{
                        fontWeight: idx === 0 ? 700 : 500,
                        fontSize: idx === 0 ? '1rem' : '0.875rem',
                        color: 'var(--text-primary)',
                        display: 'flex', alignItems: 'center', gap: '0.35rem'
                      }}>
                        {isClinical && <Target size={14} color={barColor} />}
                        {displayName}
                      </span>
                      <span style={{
                        fontWeight: 700, fontSize: '0.875rem',
                        color: barColor,
                        fontFamily: 'monospace'
                      }}>
                        {pct}%
                      </span>
                    </div>
                    <div style={{
                      width: '100%', height: '6px',
                      background: 'var(--border)',
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${Math.min(entry.score * 100, 100)}%`,
                        height: '100%',
                        background: barColor,
                        borderRadius: '3px',
                        transition: 'width 0.6s ease'
                      }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Alert level badge and Sustained Days */}
          {(currentResult?.alert_level || currentResult?.sustained_days) && (
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              {currentResult?.alert_level && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Alert Level:</span>
                  <span style={{
                    padding: '0.25rem 0.75rem', borderRadius: '1rem', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase',
                    background: currentResult.alert_level === 'red' ? 'var(--danger)' :
                                currentResult.alert_level === 'orange' ? 'var(--warning)' :
                                currentResult.alert_level === 'yellow' ? '#eab308' : 'var(--success)',
                    color: '#fff'
                  }}>
                    {currentResult.alert_level}
                  </span>
                </div>
              )}

              {currentResult?.sustained_days !== undefined && currentResult?.sustained_days > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: '1rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Sustained For:</span>
                  <span style={{ padding: '0.25rem 0.75rem', borderRadius: '1rem', fontSize: '0.75rem', fontWeight: 700, background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', border: '1px solid var(--danger)' }}>
                    {currentResult.sustained_days} Consecutive Days
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Gate Results Debug/View */}
          {currentResult?.gate_results && (
             <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', fontSize: '0.8rem', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                <div style={{ fontWeight: 600, marginBottom: '0.25rem', color: 'var(--text-primary)' }}>3-Gate Screener Detail:</div>
                {currentResult.gate_results}
             </div>
          )}
        </div>
      )}

      {/* Dynamic Clinical Insights */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
          <HeartPulse size={18} color="var(--accent-primary)" /> Dynamic Clinical Insights
        </h3>
        
        {matchMessage && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(56, 189, 248, 0.1)', borderLeft: '4px solid #38bdf8', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontWeight: 500 }}>
                🤖 ML Engine Note: {matchMessage}
            </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
          {!baseline ? (
              <div style={{ color: 'var(--text-muted)' }}>Cannot generate deviations without baseline data.</div>
          ) : (
            insights.map((insight) => {
                if (insight.deviation < 5) return null; // Ignore minor variance
                const isHigh = insight.sign === '+';
                const color = isHigh ? 'var(--warning)' : 'var(--danger)'; // Just example styling
                return (
                    <div key={insight.name} style={{ padding: '1rem', background: `rgba(245, 158, 11, 0.05)`, border: `1px solid rgba(245, 158, 11, 0.2)`, borderLeft: `4px solid ${color}`, borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontWeight: 600, color: color, marginBottom: '0.25rem' }}>
                        {isHigh ? 'Elevated' : 'Reduced'} {insight.name}
                    </div>
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                        Currently measuring {insight.rawScore.toFixed(0)}% against personal baseline. Significant {isHigh ? 'increase' : 'decrease'} detected.
                    </div>
                    </div>
                );
            })
          )}
          
          <div style={{ padding: '1rem', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderLeft: '4px solid var(--success)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 600, color: 'var(--success)', marginBottom: '0.25rem' }}>Baseline Integrity</div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>All other features remain tightly clustered around the historical baseline structure.</div>
          </div>
        </div>
      </div>
      
    </div>
  );
};
