import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged } from 'firebase/auth';
import type { User } from 'firebase/auth';
import { auth } from './firebase/config';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { PrivacyProvider } from './context/PrivacyContext';

export const App = () => {
  const [user, setUser] = useState<User | any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Loading MHealth Admin...</p>
      </div>
    );
  }

  return (
    <PrivacyProvider>
      <Router>
        <Routes>
          <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
          <Route path="/*" element={user ? <Dashboard /> : <Navigate to="/login" />} />
        </Routes>
      </Router>
    </PrivacyProvider>
  );
};

export default App;
