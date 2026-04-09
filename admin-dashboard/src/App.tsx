import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged } from 'firebase/auth';
import type { User } from 'firebase/auth';
import { auth } from './firebase/config';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { Home } from './pages/Home';
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
        <p style={{ color: 'var(--text-secondary)' }}>Loading MHealth...</p>
      </div>
    );
  }

  return (
    <PrivacyProvider>
      <Router>
        <Routes>
          {/* Default redirect to login for clinicians */}
          <Route path="/" element={<Navigate to="/login" />} />

          {/* Clinician auth */}
          <Route path="/login" element={!user ? <Login /> : <Navigate to="/dashboard" />} />

          {/* Public Home Page (Voice Assessment and overview) accessible at /about */}
          <Route path="/about" element={<Home />} />

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
