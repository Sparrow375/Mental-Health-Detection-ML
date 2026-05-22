import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged } from 'firebase/auth';
import type { User } from 'firebase/auth';
import { auth } from './firebase/config';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { LandingPage } from './pages/LandingPage';
import { PrivacyProvider } from './context/PrivacyContext.tsx';

export const App = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div className="animate-spin" style={{ width: '40px', height: '40px', border: '4px solid var(--border)', borderTopColor: 'var(--primary)', borderRadius: '50%' }} />
      </div>
    );
  }

  return (
    <PrivacyProvider>
      <Router>
        <Routes>
          {/* Landing page */}
          <Route path="/" element={<LandingPage />} />

          {/* Clinician auth */}
          <Route path="/login" element={!user ? <Login /> : <Navigate to="/dashboard" />} />

          {/* About page — also renders landing page */}
          <Route path="/about" element={<LandingPage />} />

          {/* Protected dashboard */}
          <Route
            path="/dashboard/*"
            element={user ? <Dashboard /> : <Navigate to="/login" />}
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
      </Router>
    </PrivacyProvider>
  );
};

export default App;
