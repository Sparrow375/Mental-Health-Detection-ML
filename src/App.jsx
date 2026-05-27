import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Navigation from './components/Navigation';
import Footer from './components/Footer';
import Home from './pages/Home';
import HowItWorks from './pages/HowItWorks';
import Team from './pages/Team';
import Demo from './pages/Demo';
import Download from './pages/Download';

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Home />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/team" element={<Team />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/download" element={<Download />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  // Enforce dark mode for Clinical Gold aesthetic
  if (typeof document !== 'undefined') {
    document.documentElement.classList.remove('light-mode');
  }

  return (
    <BrowserRouter>
      <Navigation />
      <main className="min-h-screen bg-primary">
        <AnimatedRoutes />
      </main>
      <Footer />
    </BrowserRouter>
  );
}
