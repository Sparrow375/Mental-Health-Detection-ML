import React, { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Download, Shield, Cloud, Brain, Cpu, Lock } from 'lucide-react';
import BaselineChart from '../components/charts/BaselineChart';

/* ── Animation Helpers ──────────────────────── */
const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 }
};

const stagger = {
  visible: { transition: { staggerChildren: 0.12 } }
};

const scaleUp = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1 }
};

interface AnimatedSectionProps {
  children: React.ReactNode;
  className?: string;
  id?: string;
}

function AnimatedSection({ children, className, id }: AnimatedSectionProps) {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <motion.section
      ref={ref}
      id={id}
      className={className}
      initial="hidden"
      animate={isInView ? 'visible' : 'hidden'}
      variants={stagger}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.section>
  );
}

interface AnimatedDivProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  variants?: Record<string, Record<string, number>>;
  delay?: number;
}

function AnimatedDiv({ children, className, style, variants: v = fadeUp, delay = 0 }: AnimatedDivProps) {
  return (
    <motion.div
      className={className}
      style={style}
      variants={v}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

/* ── Pipeline Steps Data ────────────────────── */
interface PipelineStep {
  num: number;
  title: string;
  desc: string;
}

const pipelineSteps: PipelineStep[] = [
  { num: 1, title: 'Anomaly Trigger', desc: 'System 1 escalates sustained behavioral shifts' },
  { num: 2, title: 'Model Matching', desc: 'Distance applied to Depression & BPD prototypes' },
  { num: 3, title: 'Patient Consent', desc: 'Explicit approval required to share findings' },
  { num: 4, title: 'Secure Transmit', desc: 'End-to-end encrypted transfer to care provider' },
  { num: 5, title: 'Doctor Validation', desc: 'Clinician formally reviews and validates data' },
  { num: 6, title: 'Guided Diagnosis', desc: 'Doctor safely explains insights to the patient' },
];

/* ═══════════════════════════════════════════════
   LANDING PAGE
   ═══════════════════════════════════════════════ */
export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      {/* ── NAVBAR ───────────────────────────── */}
      <nav className={`lm-nav ${scrolled ? 'scrolled' : ''}`}>
        <a href="#" className="lm-nav__logo">
          Lumen
        </a>

        <ul className="lm-nav__links">
          <li><a href="#about">About</a></li>
          <li><a href="#system-1">System-1</a></li>
          <li><a href="#system-2">System-2</a></li>
          <li><a href="#get-app">Get App</a></li>
        </ul>

        <a
          href="#"
          className="lm-nav__cta"
          onClick={(e) => { e.preventDefault(); navigate('/login'); }}
        >
          Log In
        </a>
      </nav>

      {/* ── HERO / ABOUT ────────────────────── */}
      <section className="lm-hero" id="about">
        <motion.div
          className="lm-hero__content"
          initial="hidden"
          animate="visible"
          variants={stagger}
        >
          <motion.h1 className="lm-hero__title" variants={fadeUp}>
            Early Detection.<br />
            Empowered <em>Care.</em>
          </motion.h1>

          <motion.p className="lm-hero__sub" variants={fadeUp}>
            Lumen is a clinical-grade digital health platform dedicated to the early detection and proactive monitoring of Depression and Borderline Personality Disorder (BPD). By passively analyzing everyday behavioral rhythms directly on your device, we transform subtle lifestyle shifts into actionable clinical insights. Our core mission is bridging the gap between patients and their care providers while ensuring data remains entirely private and secure.
          </motion.p>

          <motion.div variants={fadeUp}>
            <a href="#get-app" className="lm-hero__cta">
              <Download size={18} />
              Get the App
            </a>
          </motion.div>
        </motion.div>

        <motion.div
          className="lm-hero__visual"
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="lm-hero__orb-container">
            {/* Floating orbs */}
            <div className="lm-hero__orb" />
            <div className="lm-hero__orb" />
            <div className="lm-hero__orb" />

            {/* Pulsing rings */}
            <div className="lm-hero__orb-ring" />
            <div className="lm-hero__orb-ring" />
            <div className="lm-hero__orb-ring" />

            {/* Core */}
            <div className="lm-hero__orb-core">
              <Brain size={48} strokeWidth={1.5} />
            </div>
          </div>
        </motion.div>
      </section>

      {/* ── SYSTEM 1 — Anomaly Detection ───── */}
      <AnimatedSection className="lm-section lm-section--white" id="system-1">
        <div className="lm-sys1">
          <AnimatedDiv className="lm-sys1__text">
            <div className="lm-section__label">System 1</div>
            <h2 className="lm-section__title">Baseline Deviation & Early Warning</h2>
            <p className="lm-section__sub">
              Continuous, passive tracking of natural behavioral rhythms.
            </p>
            <p className="lm-sys1__desc">
              System 1 acts as a continuous early-warning engine. It silently monitors your daily digital behaviors—such as mobility, conversation frequency, and sleep windows—to establish an individualized, healthy baseline. When behaviors shift, the system tracks both the depth and velocity of those changes. To prevent false alarms triggered by everyday stress, alerts are governed by a strict 'Sustained Gate' requiring multi-day evidence. It successfully isolates unique behavioral signatures over time, identifying the gradual behavioral drifts indicative of emerging Depression or the rapid cycling variations characteristic of BPD.
            </p>
            <div className="lm-sys1__pills">
              <span className="lm-sys1__pill">Personalized Baseline</span>
              <span className="lm-sys1__pill"><Shield size={14} /> Sustained Evidence Gate</span>
              <span className="lm-sys1__pill"><Cpu size={14} /> Pattern Detection</span>
            </div>
          </AnimatedDiv>

          <AnimatedDiv className="lm-sys1__visual" delay={0.2}>
            <div className="lm-sys1__chart-wrap">
              <div className="lm-sys1__chart-header">
                <span className="lm-sys1__chart-title">Patient Behavioral Baseline</span>
                <span className="lm-sys1__chart-badge">3 Anomalies Detected</span>
              </div>
              <BaselineChart />
            </div>
          </AnimatedDiv>
        </div>
      </AnimatedSection>

      {/* ── SYSTEM 2 — Diagnostic Pipeline ──── */}
      <AnimatedSection className="lm-section lm-section--light" id="system-2">
        <div className="lm-sys2__header">
          <AnimatedDiv>
            <div className="lm-section__label">System 2</div>
          </AnimatedDiv>
          <AnimatedDiv>
            <h2 className="lm-section__title" style={{ textAlign: 'center' }}>
              Clinical Matching & Doctor Validation
            </h2>
          </AnimatedDiv>
          <AnimatedDiv>
            <p className="lm-section__sub" style={{ textAlign: 'center', margin: '0 auto' }}>
              An objective diagnostic pipeline governed entirely by patient consent and human-in-the-loop medical validation.
            </p>
          </AnimatedDiv>
          <AnimatedDiv>
            <p className="lm-sys2__desc">
              When System 1 detects a sustained behavioral anomaly, System 2 matches these shifts against validated psychiatric models for Depression and BPD. However, to prioritize patient safety and well-being, <strong>System 2 diagnostic results are strictly hidden from the user.</strong> Instead, following explicit patient consent, these highly detailed objective findings are securely transmitted to an attending doctor. The clinician formally reviews and validates the data, acting as the ultimate decision-maker. Only then does the doctor walk the patient through exactly what happened, providing a safe, definitive, and medically validated diagnosis.
            </p>
          </AnimatedDiv>
        </div>

        <div className="lm-sys2__stepper">
          {pipelineSteps.map((step, i) => (
            <motion.div
              key={step.num}
              className="lm-sys2__step"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-50px' }}
              transition={{
                duration: 0.5,
                delay: i * 0.12,
                ease: [0.16, 1, 0.3, 1]
              }}
            >
              <div className="lm-sys2__step-num">
                <span>{step.num}</span>
              </div>
              <div className="lm-sys2__step-title">{step.title}</div>
              <div className="lm-sys2__step-desc">{step.desc}</div>
            </motion.div>
          ))}
        </div>
      </AnimatedSection>

      {/* ── GET APP ─────────────────────────── */}
      <AnimatedSection className="lm-section lm-section--dark" id="get-app">
        <motion.div className="lm-app" variants={scaleUp}>
          <AnimatedDiv>
            <h2 className="lm-app__title">Download Lumen</h2>
          </AnimatedDiv>
          <AnimatedDiv>
            <p className="lm-app__sub">
              Available for Android. Privacy-first. On-device processing.
            </p>
          </AnimatedDiv>
          <AnimatedDiv variants={scaleUp}>
            <a
              href="/Lumen.apk"
              className="lm-app__download"
              download
            >
              <Download size={20} />
              Download APK
            </a>
          </AnimatedDiv>
          <AnimatedDiv>
            <div className="lm-app__pills">
              <span className="lm-app__pill">
                <Lock size={16} />
                Local Data Processing
              </span>
              <span className="lm-app__pill">
                <Shield size={16} />
                HIPAA-Aware Design
              </span>
              <span className="lm-app__pill">
                Clinician-Reviewed
              </span>
            </div>
          </AnimatedDiv>
        </motion.div>
      </AnimatedSection>

      {/* ── FOOTER ──────────────────────────── */}
      <footer className="lm-footer">
        <div className="lm-footer__inner">
          <div className="lm-footer__brand">
            Lumen
          </div>
          <div className="lm-footer__copy">
            © {new Date().getFullYear()} Lumen. All rights reserved.
          </div>
          <div className="lm-footer__tagline">
            Advanced Mental Health Diagnostics
          </div>
        </div>
      </footer>
    </>
  );
};

export default LandingPage;
