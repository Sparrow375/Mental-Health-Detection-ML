import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Dock, DockIcon } from './ui/dock';
import { Home, Info, Users, Play, Download } from 'lucide-react';
import { motion } from 'framer-motion';

const navItems = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/how-it-works', label: 'How It Works', icon: Info },
  { to: '/team', label: 'Team', icon: Users },
  { to: '/demo', label: 'Demo', icon: Play },
  { to: '/download', label: 'Beta', icon: Download },
];

export default function Navigation() {
  const { pathname } = useLocation();

  return (
    <div className="fixed bottom-6 left-0 right-0 z-50 flex justify-center pointer-events-none">
      <div className="pointer-events-auto">
        <Dock className="relative items-center bg-primary/80 backdrop-blur-xl border border-gold-border/30 rounded-full px-2 py-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.to;
            return (
              <Link 
                to={item.to} 
                key={item.to} 
                title={item.label} 
                className="relative z-10 flex items-center px-4 py-3 cursor-pointer outline-none"
              >
                {isActive && (
                  <motion.div
                    layoutId="nav-active-bg"
                    className="absolute inset-0 bg-secondary/20 rounded-full border border-secondary/30 shadow-[0_0_15px_rgba(212,175,55,0.2)]"
                    transition={{ type: "spring", stiffness: 350, damping: 25 }}
                  />
                )}
                <div className={`relative z-20 flex items-center gap-2 transition-all duration-300 ${isActive ? 'text-secondary' : 'text-text-secondary hover:text-white'}`}>
                  <Icon className="w-5 h-5" />
                  {isActive && (
                    <motion.span 
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="font-semibold text-sm whitespace-nowrap overflow-hidden"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </div>
              </Link>
            );
          })}
        </Dock>
      </div>
    </div>
  );
}
