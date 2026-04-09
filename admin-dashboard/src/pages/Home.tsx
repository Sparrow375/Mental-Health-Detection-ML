import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Brain, Shield, Zap, ChevronRight, Mic, BarChart3, Lock } from 'lucide-react';
import { VoiceAssessment } from '../components/VoiceAssessment';

const steps = [
  { icon: <Mic size={22} color="var(--accent-primary)" />, title: 'Passive Sensing', body: 'Securely collects voice features and behavior metrics from daily smartphone usage.' },
  { icon: <Brain size={22} color="var(--accent-primary)" />, title: 'Ensemble AI Analysis', body: 'Advanced ML models analyze baseline deviations to estimate mental health risks.' },
  { icon: <BarChart3 size={22} color="var(--accent-primary)" />, title: 'Actionable Insights', body: 'Clinicians receive confidence-scored risk bands to prioritize timely interventions.' },
];

const features = [
  { icon: <Zap size={20} />, label: 'Risk-averse Screening', sub: 'Optimized to minimize false negatives' },
  { icon: <Shield size={20} />, label: 'Privacy First', sub: 'On-device feature extraction, no raw storage' },
  { icon: <Lock size={20} />, label: 'Multi-modal ML', sub: 'Combines acoustic markers & behavioral data' },
];

export const Home: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-primary)' }}>

      {/* ── Sticky Nav ── */}
      <nav style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border)',
        padding: '0.875rem 2rem',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
          <Activity size={22} color="var(--accent-primary)" />
          <span style={{ fontSize: '1.1rem', fontWeight: 800, letterSpacing: '-0.03em', color: 'var(--text-primary)' }}>
            MHealth
          </span>
          <span style={{
            fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px',
            borderRadius: 999, background: 'rgba(2,132,199,0.1)', color: 'var(--accent-primary)',
            letterSpacing: '0.05em',
          }}>AI</span>
        </div>
        <button
          id="home-clinician-login-btn"
          className="btn btn-primary"
          onClick={() => navigate('/login')}
          style={{ padding: '0.5rem 1.1rem', fontSize: '0.875rem', display: 'inline-flex', gap: '0.4rem' }}
        >
          Clinician Login <ChevronRight size={15} />
        </button>
      </nav>

      {/* ── Hero ── */}
      <section style={{ maxWidth: 1100, margin: '0 auto', padding: '5rem 2rem 4rem' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0,1fr) minmax(0,420px)',
          gap: '4rem',
          alignItems: 'start',
        }}>
          {/* Left: Copy */}
          <div>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
              background: 'rgba(2,132,199,0.08)', color: 'var(--accent-primary)',
              padding: '0.4rem 1rem', borderRadius: 999,
              fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.04em',
              marginBottom: '1.5rem',
            }}>
              <Activity size={14} /> Early Risk Detection Platform
            </div>

            <h1 style={{
              fontSize: 'clamp(2rem, 4vw, 3.25rem)', fontWeight: 900,
              lineHeight: 1.08, letterSpacing: '-0.04em',
              color: 'var(--text-primary)', marginBottom: '1.25rem',
            }}>
              Early Detection of<br />
              <span style={{
                background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
                Mental Health Disorders
              </span>
            </h1>

            <p style={{
              fontSize: '1.05rem', color: 'var(--text-secondary)',
              lineHeight: 1.7, maxWidth: 480, marginBottom: '2.5rem',
            }}>
              A passive, risk-averse AI screening system that analyzes voice and behavioral indicators to flag early warning signs of depression and anxiety. Try our real-time voice assessment module.
            </p>

            {/* Feature chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '2rem' }}>
              {features.map(f => (
                <div key={f.label} style={{
                  display: 'flex', alignItems: 'center', gap: '0.625rem',
                  padding: '0.625rem 1rem',
                  background: '#fff', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-card)',
                  color: 'var(--accent-primary)',
                }}>
                  {f.icon}
                  <div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{f.label}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{f.sub}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Disclaimer */}
            <div style={{
              padding: '0.75rem 1rem',
              background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.25)',
              borderRadius: 'var(--radius-md)', fontSize: '0.8rem',
              color: 'var(--text-secondary)', maxWidth: 480,
            }}>
              ⚕️ This tool is for <strong>screening only</strong> and does not constitute a medical diagnosis. Consult a licensed professional.
            </div>
          </div>

          {/* Right: Widget */}
          <div className="glass-panel" style={{ padding: '2rem', position: 'sticky', top: '5rem' }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.625rem',
              marginBottom: '1.5rem', paddingBottom: '1rem',
              borderBottom: '1px solid var(--border)',
            }}>
              <Brain size={20} color="var(--accent-primary)" />
              <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)' }}>
                Voice Assessment
              </span>
              <span style={{
                marginLeft: 'auto', fontSize: '0.7rem', fontWeight: 700,
                padding: '2px 8px', borderRadius: 999,
                background: 'rgba(5,150,105,0.1)', color: 'var(--success)',
              }}>FREE</span>
            </div>
            <VoiceAssessment isAdmin={false} />
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section style={{
        background: '#fff', borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)', padding: '4rem 2rem',
      }}>
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>
          <h2 style={{
            textAlign: 'center', fontSize: '1.75rem', fontWeight: 800,
            letterSpacing: '-0.03em', marginBottom: '0.5rem',
          }}>How It Works</h2>
          <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '3rem' }}>
            A modular approach to early intervention.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: '2rem' }}>
            {steps.map((step, i) => (
              <div key={step.title} className="glass-panel" style={{ padding: '1.75rem' }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 'var(--radius-md)',
                  background: 'rgba(2,132,199,0.08)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center', marginBottom: '1rem',
                }}>
                  {step.icon}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <span style={{
                    fontSize: '0.7rem', fontWeight: 800, color: 'var(--accent-primary)',
                    background: 'rgba(2,132,199,0.1)', padding: '1px 7px', borderRadius: 999,
                  }}>STEP {i + 1}</span>
                  <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{step.title}</span>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.6 }}>{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Footer ── */}
      <section style={{ padding: '4rem 2rem', textAlign: 'center' }}>
        <div style={{ maxWidth: 600, margin: '0 auto' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '0.75rem' }}>
            Are you a clinician?
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', lineHeight: 1.7 }}>
            Log into the MHealth dashboard to run assessments on patients, log scores, and track outcomes over time.
          </p>
          <button
            id="home-cta-login-btn"
            className="btn btn-primary"
            onClick={() => navigate('/login')}
            style={{ fontSize: '1rem', padding: '0.875rem 2.5rem', display: 'inline-flex', gap: '0.5rem' }}
          >
            Access Clinician Dashboard <ChevronRight size={18} />
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{
        borderTop: '1px solid var(--border)', padding: '1.5rem 2rem',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        color: 'var(--text-muted)', fontSize: '0.8rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={16} color="var(--accent-primary)" />
          <span>MHealth AI · WavLM-LoRA Depression Screening</span>
        </div>
        <span>Not a diagnostic tool. For research use.</span>
      </footer>
    </div>
  );
};
