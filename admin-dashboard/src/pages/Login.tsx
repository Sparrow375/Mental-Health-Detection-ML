import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { signInWithEmailAndPassword, signInWithPopup, GoogleAuthProvider } from 'firebase/auth';
import { auth } from '../firebase/config';
import { useNavigate } from 'react-router-dom';
import { Shield, Cloud, Eye, EyeOff, Zap } from 'lucide-react';

interface BadgeData {
  icon: React.FC<{ size?: number }>;
  label: string;
  desc: string;
}

const badges: BadgeData[] = [
  {
    icon: Shield,
    label: 'HIPAA Compliant',
    desc: 'End-to-end encrypted patient data'
  },
  {
    icon: Cloud,
    label: 'Secure Cloud',
    desc: 'Enterprise-grade infrastructure'
  },
  {
    icon: Zap,
    label: 'Real-time Insights',
    desc: 'Live anomaly detection alerts'
  }
];

const tabVariants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.15 } }
};

export const Login: React.FC = () => {
  const [loginType, setLoginType] = useState<'clinician' | 'parent'>('clinician');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showClinicianPassword, setShowClinicianPassword] = useState(false);
  const [patientEmail, setPatientEmail] = useState('');
  const [patientPassword, setPatientPassword] = useState('');
  const [showPatientPassword, setShowPatientPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleClinicianLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate('/dashboard');
    } catch (err: unknown) {
      console.error('Clinician login error:', err);
      const code = (err as { code?: string }).code || '';
      console.log('Firebase error code:', code);
      if (code === 'auth/user-not-found') {
        setError('No account found with that email address.');
      } else if (code === 'auth/wrong-password' || code === 'auth/invalid-credential') {
        setError('Incorrect password. Please try again.');
      } else if (code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else if (code === 'auth/too-many-requests') {
        setError('Too many failed attempts. Please try again later.');
      } else if (code === 'auth/operation-not-allowed') {
        setError('Email/password sign-in is not enabled. Please contact the administrator.');
      } else if (code === 'auth/network-request-failed') {
        setError('Network error. Please check your internet connection.');
      } else {
        setError(`Sign in failed (${code || 'unknown'}). Please try again.`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePatientEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, patientEmail, patientPassword);
      navigate('/dashboard');
    } catch (err: unknown) {
      console.error('Patient login error:', err);
      const code = (err as { code?: string }).code || '';
      console.log('Firebase error code:', code);
      if (code === 'auth/user-not-found') {
        setError('No account found with that email address.');
      } else if (code === 'auth/wrong-password' || code === 'auth/invalid-credential') {
        setError('Incorrect password. Please try again.');
      } else if (code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else if (code === 'auth/too-many-requests') {
        setError('Too many failed attempts. Please try again later.');
      } else if (code === 'auth/operation-not-allowed') {
        setError('Email/password sign-in is not enabled. Please contact the administrator.');
      } else if (code === 'auth/network-request-failed') {
        setError('Network error. Please check your internet connection.');
      } else {
        setError(`Sign in failed (${code || 'unknown'}): ${(err as Error).message || 'Unknown error'}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleParentGoogleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      const provider = new GoogleAuthProvider();
      await signInWithPopup(auth, provider);
      navigate('/dashboard');
    } catch (err: unknown) {
      console.error('Google login error:', err);
      setError(`Google sign in failed: ${(err as Error).message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="lm-login">
      {/* ── Left: Branding Panel ─────────────── */}
      <motion.div
        className="lm-login__brand"
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="lm-login__brand-content">
          <div className="lm-login__logo">
            Lumen
          </div>
          <div className="lm-login__tagline">
            Advanced Mental Health Diagnostics Platform
          </div>

          <div className="lm-login__badges">
            {badges.map((badge, i) => (
              <motion.div
                key={badge.label}
                className="lm-login__badge"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + i * 0.12, duration: 0.5 }}
              >
                <div className="lm-login__badge-icon">
                  <badge.icon size={22} />
                </div>
                <div>
                  <div className="lm-login__badge-label">{badge.label}</div>
                  <div className="lm-login__badge-desc">{badge.desc}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ── Right: Auth Card ─────────────────── */}
      <motion.div
        className="lm-login__form-side"
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="lm-login__card">
          <h2 className="lm-login__card-title">Welcome back</h2>
          <p className="lm-login__card-sub">Sign in to access the secure portal</p>

          {/* Tab Switcher */}
          <div className="lm-login__tabs">
            <button
              type="button"
              className={`lm-login__tab ${loginType === 'clinician' ? 'lm-login__tab--active' : ''}`}
              onClick={() => { setLoginType('clinician'); setError(''); }}
            >
              Clinician
            </button>
            <button
              type="button"
              className={`lm-login__tab ${loginType === 'parent' ? 'lm-login__tab--active' : ''}`}
              onClick={() => { setLoginType('parent'); setError(''); }}
            >
              Patient / Parent
            </button>

            {/* Animated underline indicator */}
            <motion.div
              className="lm-login__tab-indicator"
              layoutId="tab-indicator"
              style={{
                width: '50%',
                left: loginType === 'clinician' ? '0%' : '50%'
              }}
              transition={{ type: 'spring', stiffness: 400, damping: 35 }}
            />
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                className="lm-login__error"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Tab Content */}
          <AnimatePresence mode="wait">
            {loginType === 'clinician' ? (
              <motion.div
                key="clinician"
                variants={tabVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                <form onSubmit={handleClinicianLogin}>
                  <div className="lm-login__field">
                    <label className="lm-login__label" htmlFor="clinician-email">Email Address</label>
                    <input
                      id="clinician-email"
                      type="email"
                      className="lm-login__input"
                      placeholder="clinician@hospital.org"
                      required
                      value={email}
                      onChange={e => setEmail(e.target.value)}
                    />
                  </div>
                  <div className="lm-login__field">
                    <label className="lm-login__label" htmlFor="clinician-password">Password</label>
                    <div className="lm-login__password-wrap">
                      <input
                        id="clinician-password"
                        type={showClinicianPassword ? 'text' : 'password'}
                        className="lm-login__input"
                        placeholder="Enter your password"
                        required
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="lm-login__password-toggle"
                        onClick={() => setShowClinicianPassword(v => !v)}
                        aria-label={showClinicianPassword ? 'Hide password' : 'Show password'}
                      >
                        {showClinicianPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>
                  <button type="submit" className="lm-login__submit" disabled={loading}>
                    {loading ? 'Authenticating...' : 'Log In'}
                  </button>
                </form>
              </motion.div>
            ) : (
              <motion.div
                key="parent"
                variants={tabVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                {/* Email/Password login for patients */}
                <form onSubmit={handlePatientEmailLogin}>
                  <div className="lm-login__field">
                    <label className="lm-login__label" htmlFor="patient-email">Email Address</label>
                    <input
                      id="patient-email"
                      type="email"
                      className="lm-login__input"
                      placeholder="you@email.com"
                      required
                      value={patientEmail}
                      onChange={e => setPatientEmail(e.target.value)}
                    />
                  </div>
                  <div className="lm-login__field">
                    <label className="lm-login__label" htmlFor="patient-password">Password</label>
                    <div className="lm-login__password-wrap">
                      <input
                        id="patient-password"
                        type={showPatientPassword ? 'text' : 'password'}
                        className="lm-login__input"
                        placeholder="Enter your password"
                        required
                        value={patientPassword}
                        onChange={e => setPatientPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="lm-login__password-toggle"
                        onClick={() => setShowPatientPassword(v => !v)}
                        aria-label={showPatientPassword ? 'Hide password' : 'Show password'}
                      >
                        {showPatientPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>
                  <button type="submit" className="lm-login__submit" disabled={loading}>
                    {loading ? 'Authenticating...' : 'Log In'}
                  </button>
                </form>

                {/* Divider */}
                <div className="lm-login__divider">
                  <span>or</span>
                </div>

                {/* Google login */}
                <button
                  type="button"
                  className="lm-login__google"
                  onClick={handleParentGoogleLogin}
                  disabled={loading}
                >
                  <img
                    src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
                    alt="Google"
                  />
                  Continue with Google
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};
