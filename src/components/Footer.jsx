import { Link } from 'react-router-dom';

const links = [
  { to: '/', label: 'Home' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/team', label: 'Team' },
  { to: '/demo', label: 'Demo' },
  { to: '/download', label: 'Beta Testing' },
];

export default function Footer() {
  return (
    <footer className="bg-bg-dark text-text-light">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="flex flex-col md:flex-row justify-between items-start gap-12">
          <div className="max-w-sm">
            <div className="flex items-center gap-2 mb-4">

              <span className="text-lg font-bold text-text-primary">Lumen.</span>
            </div>
            <p className="text-sm text-text-light/60 leading-relaxed">
              Passive behavioral anomaly detection for early mental health risk screening. Made with care for the people who need it most.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-light/40 mb-1">Pages</span>
            {links.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className="text-sm text-text-light/60 hover:text-text-primary transition-colors"
              >
                {link.label}
              </Link>
            ))}
          </div>

          <div className="flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-light/40 mb-1">Privacy</span>
            <p className="text-sm text-text-light/60 max-w-xs">
              All behavioral data is processed entirely on-device. Your data never leaves your phone. No cloud uploads. No third-party access.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-light/40 mb-1">Contact</span>
            <a
              href="mailto:lumen.mh.in@gmail.com"
              className="flex items-center gap-2 text-sm text-text-light/60 hover:text-text-primary transition-colors"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
              </svg>
              lumen.mh.in@gmail.com
            </a>
            <a
              href="https://www.instagram.com/lumen.mh/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-text-light/60 hover:text-text-primary transition-colors"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/>
              </svg>
              lumen.mh
            </a>
          </div>
        </div>

        <div className="mt-16 pt-8 border-t border-[var(--color-glass-border)] flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-text-light/40">
            &copy; {new Date().getFullYear()} Lumen. Research project &mdash; not a medical device.
          </p>
          <p className="text-xs text-text-light/30">
            Built for early detection, not diagnosis.
          </p>
        </div>
      </div>
    </footer>
  );
}
