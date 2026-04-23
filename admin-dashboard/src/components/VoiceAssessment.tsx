import React, { useState, useRef, useEffect } from 'react';
import {
  Mic, Upload, Square, Loader2, Brain, AlertCircle,
  CheckCircle, MicOff,
} from 'lucide-react';
import { collection, addDoc, serverTimestamp, getDocs, query, orderBy, limit } from 'firebase/firestore';
import { db, auth } from '../firebase/config';

const HF_API_URL = (import.meta.env.VITE_HF_API_URL as string) || '';
const MAX_RECORD_SECONDS = 300; // 5 minutes

interface VoiceAssessmentProps {
  isAdmin?: boolean;
}

interface PredictionResult {
  prediction: string;
  probability: number;
  confidence: number;
  n_chunks: number;
}

type Tab = 'upload' | 'record';
type Status = 'idle' | 'wakeup' | 'recording' | 'processing' | 'done' | 'error';

const fmt = (s: number) =>
  `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

export const VoiceAssessment: React.FC<VoiceAssessmentProps> = ({ isAdmin = false }) => {
  const [tab, setTab] = useState<Tab>('upload');
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [patientName, setPatientName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordSecs, setRecordSecs] = useState(0);
  const [wakeupSecs, setWakeupSecs] = useState(120);
  const [micDenied, setMicDenied] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [cancelWakeup, setCancelWakeup] = useState(false);

  // ── Past assessments (read-only) ──
  interface AssessmentRecord {
    id: string;
    patient_name: string;
    prediction: string;
    probability: number;
    confidence: number;
    n_chunks: number;
    clinician_id: string;
    timestamp: { seconds: number; nanoseconds: number } | null;
  }
  const [historyRecords, setHistoryRecords] = useState<AssessmentRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const q = query(
          collection(db, 'voice_assessments'),
          orderBy('timestamp', 'desc'),
          limit(50)
        );
        const snap = await getDocs(q);
        setHistoryRecords(
          snap.docs.map(d => ({ id: d.id, ...(d.data() as Omit<AssessmentRecord, 'id'>) }))
        );
      } catch (e) {
        console.warn('Could not fetch voice assessment history:', e);
      } finally {
        setHistoryLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const mrRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Guard ref: prevents concurrent analyze() calls (including React StrictMode double-invocations)
  const isAnalyzingRef = useRef(false);

  // ── API: check if HF is alive ──
  const isApiAlive = async () => {
    try {
      const res = await fetch(`${HF_API_URL}/api/health`, {
        signal: AbortSignal.timeout(5000),
        cache: 'no-cache',
      });
      console.log('[VoiceAssessment] API health check:', res.ok ? 'OK' : 'FAILED', res.status);
      return res.ok;
    } catch (e) {
      console.warn('[VoiceAssessment] API health check failed:', e);
      return false;
    }
  };

  const pingUntilAlive = async (): Promise<boolean> => {
    setStatus('wakeup');
    setCancelWakeup(false);
    let countdown = 120;
    setWakeupSecs(countdown);
    countdownRef.current = setInterval(() => {
      countdown -= 1;
      setWakeupSecs(countdown);
      if (countdown <= 0) clearInterval(countdownRef.current!);
    }, 1000);
    for (let i = 0; i < 12; i++) {
      if (cancelWakeup) {
        clearInterval(countdownRef.current!);
        return false;
      }
      const alive = await isApiAlive();
      if (alive) { clearInterval(countdownRef.current!); return true; }
      await new Promise(r => setTimeout(r, 10_000));
    }
    clearInterval(countdownRef.current!);
    return false;
  };

  // ── Core: analyze ──
  const analyze = async (audio: File | Blob, filename: string) => {
    // Synchronous guard: reject concurrent calls immediately (before any await)
    if (isAnalyzingRef.current) return;
    isAnalyzingRef.current = true;

    console.log('[VoiceAssessment] Starting analysis for:', filename);

    if (!HF_API_URL) {
      setError('API URL not configured. Set VITE_HF_API_URL in your .env.local file.');
      setStatus('error');
      isAnalyzingRef.current = false;
      return;
    }

    // Set a loading status synchronously so the UI disables the button immediately
    setStatus('processing');

    try {
      console.log('[VoiceAssessment] Checking API health...');
      const alive = await isApiAlive();
      if (!alive) {
        console.log('[VoiceAssessment] API not alive, attempting wake-up...');
        const woke = await pingUntilAlive();
        if (!woke) {
          setError('AI server timed out during wake-up. Please try again in a few minutes.');
          setStatus('error');
          isAnalyzingRef.current = false;
          return;
        }
      }

      console.log('[VoiceAssessment] Sending audio to API...');
      const form = new FormData();
      form.append('file', audio, filename);
      const res = await fetch(`${HF_API_URL}/api/predict`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(180000), // 3 minute timeout for inference (HF can be slow)
      });
      console.log('[VoiceAssessment] API response status:', res.status);
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const data: PredictionResult = await res.json();
      console.log('[VoiceAssessment] Analysis complete:', data);

      // Save to Firestore BEFORE updating UI state to prevent double-save
      // (state updates are batched; saving first ensures exactly-once semantics)
      if (isAdmin && patientName.trim()) {
        await addDoc(collection(db, 'voice_assessments'), {
          patient_name: patientName.trim(),
          prediction: data.prediction,
          probability: data.probability,
          confidence: data.confidence,
          n_chunks: data.n_chunks,
          clinician_id: auth.currentUser?.uid ?? 'unknown',
          timestamp: serverTimestamp(),
        });
      }

      setResult(data);
      setStatus('done');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed. Please try again.');
      setStatus('error');
    } finally {
      isAnalyzingRef.current = false;
    }
  };

  // ── File upload ──
  const handleFileSelect = (f: File) => {
    setFile(f); setResult(null); setError('');
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFileSelect(f);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('audio/')) handleFileSelect(f);
  };

  // ── Recording ──
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      chunksRef.current = [];
      mr.ondataavailable = e => chunksRef.current.push(e.data);
      mr.onstop = () => {
        setRecordedBlob(new Blob(chunksRef.current, { type: 'audio/webm' }));
        stream.getTracks().forEach(t => t.stop());
      };
      mr.start(1000);
      mrRef.current = mr;
      setStatus('recording');
      setRecordSecs(0);
      timerRef.current = setInterval(() => {
        setRecordSecs(s => {
          if (s + 1 >= MAX_RECORD_SECONDS) { stopRecording(); }
          return s + 1;
        });
      }, 1000);
    } catch (e: unknown) {
      const err = e as DOMException;
      if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') {
        setMicDenied(true); setTab('upload');
      } else {
        setError('Microphone error: ' + err.message);
      }
    }
  };

  const stopRecording = () => {
    if (mrRef.current?.state !== 'inactive') mrRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
    setStatus('idle');
  };

  const reset = () => {
    isAnalyzingRef.current = false;
    if (countdownRef.current) clearInterval(countdownRef.current);
    setCancelWakeup(true);
    setStatus('idle'); setResult(null); setError('');
    setFile(null); setRecordedBlob(null); setRecordSecs(0); setPatientName('');
  };

  const canAnalyze =
    status === 'idle' &&
    ((tab === 'upload' && file !== null) || (tab === 'record' && recordedBlob !== null)) &&
    (!isAdmin || patientName.trim().length > 0);

  const isDepressed = result?.prediction === 'Depressed';
  const prob = result ? Math.round(result.probability * 100) : 0;

  /* ── Shared styles ── */
  const s = {
    card: { padding: '1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)', marginBottom: '1.5rem' } as React.CSSProperties,
    row: { display: 'flex', alignItems: 'center', gap: '0.75rem' } as React.CSSProperties,
  };

  return (
    <div style={{ fontFamily: "'Inter', sans-serif" }}>
      <style>{`@media (max-width: 900px) { .va-layout { flex-direction: column !important; } }`}</style>
      <div className="va-layout" style={{ display: 'flex', flexDirection: 'row', gap: '1.5rem', alignItems: 'flex-start' }}>

        {/* ── LEFT COLUMN: Analysis Form ── */}
        <div style={{ minWidth: '520px', maxWidth: '560px', flex: '0 0 auto' }}>

      {/* ── Admin: patient name ── */}
      {isAdmin && (
        <div className="input-group">
          <label className="input-label" htmlFor="voice-patient-name">Patient Name *</label>
          <input
            id="voice-patient-name"
            className="input-field"
            type="text"
            placeholder="Enter patient name to enable analysis…"
            value={patientName}
            onChange={e => setPatientName(e.target.value)}
            disabled={status === 'processing' || status === 'wakeup'}
          />
        </div>
      )}

      {/* ── Wakeup State ── */}
      {status === 'wakeup' && (
        <div style={{ ...s.card, textAlign: 'center', background: 'rgba(2,132,199,0.04)', borderColor: 'rgba(2,132,199,0.25)' }}>
          <Loader2 size={36} color="var(--accent-primary)" style={{ marginBottom: '1rem', animation: 'spin 1.2s linear infinite' }} />
          <p style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '0.25rem' }}>Booting AI Engine…</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem' }}>
            The AI is waking from sleep. Usually takes 1–2 minutes.
          </p>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent-primary)', marginBottom: '0.75rem' }}>
            {fmt(wakeupSecs)}
          </div>
          <div style={{ height: 6, background: 'var(--border)', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{ height: '100%', borderRadius: 999, background: 'linear-gradient(90deg,var(--accent-primary),var(--accent-secondary))', width: `${((120 - wakeupSecs) / 120) * 100}%`, transition: 'width 1s linear' }} />
          </div>
          <button className="btn btn-secondary" onClick={reset} style={{ marginTop: '1rem', width: '100%' }}>
            Cancel
          </button>
        </div>
      )}

      {/* ── Processing ── */}
      {status === 'processing' && (
        <div style={{ ...s.card, textAlign: 'center', background: 'rgba(2,132,199,0.04)', borderColor: 'rgba(2,132,199,0.25)' }}>
          <Loader2 size={36} color="var(--accent-primary)" style={{ marginBottom: '1rem', animation: 'spin 1.2s linear infinite' }} />
          <p style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '0.25rem' }}>Analyzing Voice Patterns…</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem' }}>Running WavLM across audio chunks</p>
          <button className="btn btn-secondary" onClick={reset} style={{ width: '100%' }}>
            Cancel
          </button>
        </div>
      )}

      {/* ── Results ── */}
      {status === 'done' && result && (
        <div style={{ animation: 'fadeIn 0.4s ease' }}>
          <div style={{
            ...s.card,
            textAlign: 'center',
            border: `2px solid ${isDepressed ? 'var(--danger)' : 'var(--success)'}`,
            background: isDepressed ? 'rgba(220,38,38,0.04)' : 'rgba(5,150,105,0.04)',
          }}>
            {isDepressed
              ? <AlertCircle size={44} color="var(--danger)" style={{ marginBottom: '0.75rem' }} />
              : <CheckCircle size={44} color="var(--success)" style={{ marginBottom: '0.75rem' }} />}
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: isDepressed ? 'var(--danger)' : 'var(--success)' }}>
              {isDepressed ? 'Signs Detected' : 'No Signs Detected'}
            </div>
            <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.875rem' }}>
              {result.prediction} · {result.n_chunks} chunks analyzed
            </div>
          </div>

          {/* Probability bar */}
          <div style={{ ...s.card }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
              <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Depression Probability</span>
              <span style={{ fontWeight: 800, color: isDepressed ? 'var(--danger)' : 'var(--success)' }}>{prob}%</span>
            </div>
            <div style={{ height: 10, background: 'var(--bg-primary)', borderRadius: 999, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 999,
                background: isDepressed ? 'var(--danger)' : 'var(--success)',
                width: `${prob}%`, transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Decision threshold: 40%</span>
              <span>Confidence: {Math.round(result.confidence * 100)}%</span>
            </div>
          </div>

          {!isAdmin && (
            <div style={{ padding: '0.75rem 1rem', background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 'var(--radius-md)', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              This is an AI screening tool, <strong>not a clinical diagnosis</strong>. Please consult a licensed healthcare professional.
            </div>
          )}

          <button id="voice-reset-btn" className="btn btn-secondary" onClick={reset} style={{ width: '100%' }}>
            Analyze Another Recording
          </button>
        </div>
      )}

      {/* ── Error ── */}
      {status === 'error' && (
        <div style={{ ...s.card, background: 'rgba(220,38,38,0.04)', borderColor: 'rgba(220,38,38,0.3)', color: 'var(--danger)' }}>
          <div style={s.row}><AlertCircle size={18} /><strong>Error</strong></div>
          <p style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>{error}</p>
          <button className="btn btn-secondary" onClick={reset} style={{ marginTop: '0.75rem', width: '100%' }}>Try Again</button>
        </div>
      )}

      {/* ── Main Input (idle / recording) ── */}
      {(status === 'idle' || status === 'recording') && (
        <>
          {/* Tab Switcher */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem', background: 'var(--bg-primary)', padding: '4px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
            {(['upload', 'record'] as Tab[]).map(t => {
              const active = tab === t;
              const disabled = t === 'record' && micDenied;
              return (
                <button
                  key={t}
                  id={`voice-tab-${t}`}
                  onClick={() => { if (!disabled) setTab(t); }}
                  style={{
                    flex: 1, padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)',
                    background: active ? '#fff' : 'transparent',
                    color: disabled ? 'var(--text-muted)' : active ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontWeight: active ? 600 : 400, cursor: disabled ? 'not-allowed' : 'pointer',
                    border: active ? '1px solid var(--border)' : 'none',
                    boxShadow: active ? 'var(--shadow-card)' : 'none',
                    transition: 'all 0.2s', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: '0.4rem', fontSize: '0.875rem',
                  }}
                >
                  {t === 'upload' ? <Upload size={15} /> : micDenied ? <MicOff size={15} /> : <Mic size={15} />}
                  {t === 'upload' ? 'Upload File' : micDenied ? 'Mic Disabled' : 'Live Record'}
                </button>
              );
            })}
          </div>

          {/* Upload Zone */}
          {tab === 'upload' && (
            <div
              id="voice-upload-zone"
              onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? 'var(--accent-primary)' : file ? 'var(--success)' : 'var(--border)'}`,
                borderRadius: 'var(--radius-lg)', padding: '2.5rem 1.5rem', textAlign: 'center',
                cursor: 'pointer', marginBottom: '1.25rem',
                background: isDragging ? 'rgba(2,132,199,0.04)' : file ? 'rgba(5,150,105,0.04)' : 'var(--bg-primary)',
                transition: 'all 0.2s',
              }}
            >
              {file ? (
                <>
                  <CheckCircle size={36} color="var(--success)" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontWeight: 600 }}>{file.name}</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB — click to change
                  </p>
                </>
              ) : (
                <>
                  <Upload size={36} color="var(--text-muted)" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontWeight: 600 }}>Drop audio file here</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    or click to browse · MP3, WAV, FLAC, M4A, OGG
                  </p>
                </>
              )}
              <input id="voice-file-input" ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileChange} style={{ display: 'none' }} />
            </div>
          )}

          {/* Record Zone */}
          {tab === 'record' && !micDenied && (
            <div style={{ textAlign: 'center', padding: '1.5rem', marginBottom: '1.25rem' }}>
              {status === 'recording' ? (
                <>
                  <div style={{
                    width: 80, height: 80, borderRadius: '50%', margin: '0 auto 1rem',
                    background: 'rgba(220,38,38,0.1)', border: '3px solid var(--danger)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    animation: 'pulse 1.5s ease-in-out infinite',
                  }}>
                    <Mic size={32} color="var(--danger)" />
                  </div>
                  <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--danger)', marginBottom: '0.25rem' }}>
                    {fmt(recordSecs)}
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
                    Recording… (max {fmt(MAX_RECORD_SECONDS)})
                  </p>
                  <button id="voice-stop-btn" className="btn" onClick={stopRecording}
                    style={{ background: 'var(--danger)', color: '#fff', gap: '0.5rem', display: 'inline-flex' }}>
                    <Square size={16} /> Stop Recording
                  </button>
                </>
              ) : recordedBlob ? (
                <>
                  <CheckCircle size={40} color="var(--success)" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontWeight: 600 }}>Recording ready ({fmt(recordSecs)})</p>
                  <button id="voice-rerecord-btn" className="btn btn-secondary"
                    onClick={() => { setRecordedBlob(null); setRecordSecs(0); }}
                    style={{ marginTop: '0.75rem' }}>
                    Re-record
                  </button>
                </>
              ) : (
                <>
                  <div style={{
                    width: 80, height: 80, borderRadius: '50%', margin: '0 auto 1rem',
                    background: 'rgba(2,132,199,0.08)', border: '3px solid var(--accent-primary)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Mic size={32} color="var(--accent-primary)" />
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1.25rem' }}>
                    Press to record up to 5 minutes
                  </p>
                  <button id="voice-start-btn" className="btn btn-primary" onClick={startRecording}
                    style={{ display: 'inline-flex', gap: '0.5rem' }}>
                    <Mic size={16} /> Start Recording
                  </button>
                </>
              )}
            </div>
          )}

          {/* Analyze Button */}
          <button
            id="voice-analyze-btn"
            className="btn btn-primary"
            onClick={() => {
              if (tab === 'upload' && file) analyze(file, file.name);
              else if (tab === 'record' && recordedBlob) analyze(recordedBlob, 'recording.webm');
            }}
            disabled={!canAnalyze}
            style={{ width: '100%', opacity: canAnalyze ? 1 : 0.45, cursor: canAnalyze ? 'pointer' : 'not-allowed' }}
          >
            <Brain size={18} />
            {isAdmin && !patientName.trim()
              ? 'Enter patient name to analyze'
              : 'Analyze Voice'}
          </button>
        </>
      )}
      </div>{/* end left column */}

        {/* ── RIGHT COLUMN: Past Assessments ── */}
      {isAdmin && (
        <div style={{ flex: 1, minWidth: '320px' }}>
          <div style={{ ...s.card, marginBottom: '0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
              <Brain size={18} color="var(--accent-primary)" />
              <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>Past Assessments</span>
            </div>

            {/* Loading */}
            {historyLoading && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1rem 0', color: 'var(--text-secondary)' }}>
                <Loader2 size={18} style={{ animation: 'spin 1.2s linear infinite' }} />
                <span style={{ fontSize: '0.875rem' }}>Loading history…</span>
              </div>
            )}

            {/* Empty */}
            {!historyLoading && historyRecords.length === 0 && (
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>No assessments recorded yet.</p>
            )}

            {/* List */}
            {!historyLoading && historyRecords.length > 0 && (
              <div style={{ maxHeight: '500px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {historyRecords.map(rec => {
                  const isDepr = rec.prediction === 'Depressed';
                  const pct = Math.round((rec.probability ?? 0) * 100);
                  const ts = rec.timestamp?.seconds
                    ? new Date(rec.timestamp.seconds * 1000).toLocaleString()
                    : '—';
                  return (
                    <div
                      key={rec.id}
                      style={{
                        padding: '0.875rem 1rem',
                        borderRadius: 'var(--radius-md)',
                        border: `1px solid ${isDepr ? 'rgba(220,38,38,0.25)' : 'rgba(5,150,105,0.25)'}`,
                        background: isDepr ? 'rgba(220,38,38,0.03)' : 'rgba(5,150,105,0.03)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '1rem',
                        flexWrap: 'wrap',
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', flex: 1, minWidth: 0 }}>
                        <span style={{ fontWeight: 600, fontSize: '0.9rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {rec.patient_name}
                        </span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{ts}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
                        <span style={{
                          fontSize: '0.8rem', fontWeight: 700, fontFamily: 'monospace',
                          color: isDepr ? 'var(--danger)' : 'var(--success)',
                        }}>
                          {pct}%
                        </span>
                        <span style={{
                          padding: '0.2rem 0.6rem',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          background: isDepr ? 'rgba(220,38,38,0.1)' : 'rgba(5,150,105,0.1)',
                          color: isDepr ? 'var(--danger)' : 'var(--success)',
                        }}>
                          {rec.prediction}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
      </div>{/* end outer flex row */}
    </div>
  );
};
