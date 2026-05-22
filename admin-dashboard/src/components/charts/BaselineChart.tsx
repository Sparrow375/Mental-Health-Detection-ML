import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

/**
 * BaselineChart — SVG animated line chart showing behavioral baseline
 * with deviation spikes. Wobble/jiggle on hover via Framer Motion.
 */

interface ChartPoint {
  x: number;
  y: number;
}

interface SpikeZone {
  start: number;
  end: number;
  label: string;
}

// Generate deterministic "baseline" data with deviation spikes
function generateChartData(): ChartPoint[] {
  const points: ChartPoint[] = [];
  const baselineValue = 50;

  for (let i = 0; i <= 60; i++) {
    let value = baselineValue + Math.sin(i * 0.3) * 5 + Math.cos(i * 0.7) * 3;

    // Inject deviation spikes at specific points
    if (i >= 18 && i <= 22) {
      value += 25 + Math.sin((i - 18) * 1.2) * 10; // Large spike
    }
    if (i >= 38 && i <= 40) {
      value += 18 + Math.sin((i - 38) * 1.5) * 6; // Medium spike
    }
    if (i >= 52 && i <= 55) {
      value += 22 + Math.cos((i - 52) * 0.8) * 8; // Another spike
    }

    points.push({ x: i, y: Math.max(5, Math.min(95, value)) });
  }
  return points;
}

// Generate EWMA velocity line
function generateEWMA(data: ChartPoint[]): ChartPoint[] {
  const alpha = 0.15;
  const ewma: ChartPoint[] = [];
  let prev = data[0].y;

  for (let i = 0; i < data.length; i++) {
    prev = alpha * data[i].y + (1 - alpha) * prev;
    ewma.push({ x: data[i].x, y: prev });
  }
  return ewma;
}

function pointsToPath(points: ChartPoint[], width: number, height: number, padding: number): string {
  const xScale = (width - padding * 2) / 60;
  const yScale = (height - padding * 2) / 100;

  return points.map((p, i) => {
    const x = padding + p.x * xScale;
    const y = height - padding - p.y * yScale;
    return `${i === 0 ? 'M' : 'L'}${x},${y}`;
  }).join(' ');
}

function pointsToAreaPath(points: ChartPoint[], width: number, height: number, padding: number): string {
  const xScale = (width - padding * 2) / 60;
  const yScale = (height - padding * 2) / 100;

  const linePath = points.map((p, i) => {
    const x = padding + p.x * xScale;
    const y = height - padding - p.y * yScale;
    return `${i === 0 ? 'M' : 'L'}${x},${y}`;
  }).join(' ');

  const lastX = padding + points[points.length - 1].x * xScale;
  const firstX = padding + points[0].x * xScale;
  const bottom = height - padding;

  return `${linePath} L${lastX},${bottom} L${firstX},${bottom} Z`;
}

export default function BaselineChart() {
  const width = 560;
  const height = 280;
  const padding = 30;

  const data = useMemo(() => generateChartData(), []);
  const ewma = useMemo(() => generateEWMA(data), [data]);

  const baselinePath = pointsToPath(data, width, height, padding);
  const baselineArea = pointsToAreaPath(data, width, height, padding);
  const ewmaPath = pointsToPath(ewma, width, height, padding);

  // Threshold line y position
  const thresholdY = height - padding - (65 * (height - padding * 2) / 100);

  // Spike zone markers
  const xScale = (width - padding * 2) / 60;
  const spikeZones: SpikeZone[] = [
    { start: 18, end: 22, label: 'Sustained\nDeviation' },
    { start: 38, end: 40, label: 'Spike' },
    { start: 52, end: 55, label: '4-Day\nDeviation' },
  ];

  return (
    <motion.div
      whileHover={{
        x: [0, -3, 4, -4, 3, -2, 1, 0],
        transition: {
          duration: 0.6,
          ease: 'easeInOut',
          times: [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1]
        }
      }}
      style={{ width: '100%' }}
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: 'block' }}
      >
        <defs>
          <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0D7377" stopOpacity={0.20} />
            <stop offset="100%" stopColor="#0D7377" stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="spikeGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#E8614D" stopOpacity={0.15} />
            <stop offset="100%" stopColor="#E8614D" stopOpacity={0.02} />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[20, 40, 60, 80].map(val => {
          const y = height - padding - (val * (height - padding * 2) / 100);
          return (
            <g key={val}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="#E8E6E1"
                strokeWidth="1"
                strokeDasharray="4,4"
              />
              <text
                x={padding - 8}
                y={y + 4}
                textAnchor="end"
                fill="#8896A6"
                fontSize="10"
                fontFamily="Inter, sans-serif"
              >
                {val}
              </text>
            </g>
          );
        })}

        {/* X-axis labels */}
        {[0, 10, 20, 30, 40, 50, 60].map(day => (
          <text
            key={day}
            x={padding + day * xScale}
            y={height - 8}
            textAnchor="middle"
            fill="#8896A6"
            fontSize="10"
            fontFamily="Inter, sans-serif"
          >
            D{day}
          </text>
        ))}

        {/* Threshold line */}
        <line
          x1={padding}
          y1={thresholdY}
          x2={width - padding}
          y2={thresholdY}
          stroke="#E8614D"
          strokeWidth="1.5"
          strokeDasharray="6,4"
          opacity="0.6"
        />
        <text
          x={width - padding + 4}
          y={thresholdY + 4}
          fill="#E8614D"
          fontSize="9"
          fontWeight="600"
          fontFamily="Inter, sans-serif"
        >
          Alert
        </text>

        {/* Spike zone highlights */}
        {spikeZones.map((zone, i) => (
          <motion.rect
            key={i}
            x={padding + zone.start * xScale}
            y={padding}
            width={(zone.end - zone.start) * xScale}
            height={height - padding * 2}
            fill="url(#spikeGrad)"
            rx="4"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 1.2 + (i * 0.2) }}
          />
        ))}

        {/* Baseline area fill */}
        <motion.path
          d={baselineArea}
          fill="url(#baselineGrad)"
          initial={{ opacity: 0, clipPath: 'inset(0 100% 0 0)' }}
          whileInView={{ opacity: 1, clipPath: 'inset(0 0% 0 0)' }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 1.5, ease: 'easeInOut', delay: 0.2 }}
        />

        {/* EWMA velocity line */}
        <motion.path
          d={ewmaPath}
          fill="none"
          stroke="#D4A853"
          strokeWidth="2"
          strokeDasharray="5,3"
          opacity="0.7"
          initial={{ pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 1.5, ease: 'easeInOut', delay: 0.1 }}
        />

        {/* Main baseline path */}
        <motion.path
          d={baselinePath}
          fill="none"
          stroke="#0D7377"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 1.5, ease: 'easeInOut', delay: 0 }}
        />

        {/* Spike peak dots */}
        {[20, 39, 53].map((peakX, i) => {
          const peakPoint = data[peakX];
          const cx = padding + peakPoint.x * xScale;
          const cy = height - padding - (peakPoint.y * (height - padding * 2) / 100);
          return (
            <motion.g 
              key={i}
              initial={{ scale: 0, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 1.4 + (i * 0.15), type: 'spring' }}
            >
              <circle cx={cx} cy={cy} r="6" fill="#E8614D" opacity="0.2" />
              <circle cx={cx} cy={cy} r="3.5" fill="#E8614D" />
              <circle cx={cx} cy={cy} r="2" fill="white" />
            </motion.g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: '20px',
        marginTop: '12px',
        paddingLeft: '30px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '3px', background: '#0D7377', borderRadius: '2px' }} />
          <span style={{ fontSize: '0.75rem', color: '#8896A6', fontWeight: 500 }}>Personalized Baseline</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '3px', background: '#D4A853', borderRadius: '2px', opacity: 0.7 }} />
          <span style={{ fontSize: '0.75rem', color: '#8896A6', fontWeight: 500 }}>Continuous Monitoring</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '8px', height: '8px', background: '#E8614D', borderRadius: '50%' }} />
          <span style={{ fontSize: '0.75rem', color: '#8896A6', fontWeight: 500 }}>Sustained Anomaly</span>
        </div>
      </div>
    </motion.div>
  );
}
