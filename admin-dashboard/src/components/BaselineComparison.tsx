import React from 'react';

interface ArcProgressRingProps {
  value: number;
  baseline: number;
  label: string;
  unit: string;
  max?: number;
}

export const ArcProgressRing: React.FC<ArcProgressRingProps> = ({
  value,
  baseline,
  label,
  unit,
  max = baseline * 1.5,
}) => {
  const size = 120;
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * Math.PI;

  const percent = Math.min(Math.max(value / max, 0), 1);
  const strokeDashoffset = circumference - (percent * circumference);

  // Single unified color based on how close to baseline
  const getColor = () => {
    if (baseline === 0) return 'var(--accent-primary)';
    const ratio = value / baseline;
    // Green when close to baseline (0.8-1.2), blue otherwise
    if (ratio >= 0.8 && ratio <= 1.2) return 'var(--success)';
    return 'var(--accent-primary)';
  };

  const color = getColor();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '1.5rem 1rem',
      background: 'var(--bg-card)',
      borderRadius: 'var(--radius-lg)',
      border: '1px solid var(--border)',
    }}>
      <svg width={size} height={size / 2 + 20} style={{ transform: 'rotate(180deg)' }}>
        <path
          d={`M ${strokeWidth} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        <path
          d={`M ${strokeWidth} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
      </svg>

      <div style={{
        position: 'absolute',
        top: '3.5rem',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          {value.toFixed(1)}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{unit}</div>
      </div>

      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{label}</div>
      </div>
    </div>
  );
};
