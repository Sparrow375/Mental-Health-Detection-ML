import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';

export interface BaselineMetric {
  label: string;
  current: number;
  baseline: number;
  unit: string;
  invertGood?: boolean;
}

interface ChartDataPoint {
  name: string;
  baseline: number;
  current: number;
  deviation: number;
  deviationPercent: number;
  invertGood?: boolean;
}

const BaselineLineGraph: React.FC<{ metrics: BaselineMetric[] }> = ({ metrics }) => {
  // Transform metrics into chart data
  const chartData: ChartDataPoint[] = metrics.map((metric) => {
    const deviation = metric.current - metric.baseline;
    const deviationPercent = metric.baseline !== 0 ? (deviation / metric.baseline) * 100 : 0;
    return {
      name: metric.label,
      baseline: metric.baseline,
      current: metric.current,
      deviation,
      deviationPercent,
      invertGood: metric.invertGood,
    };
  });

  // Calculate Y-axis domain
  const allValues = chartData.flatMap(d => [d.baseline, d.current]);
  const minValue = Math.min(...allValues) * 0.9;
  const maxValue = Math.max(...allValues) * 1.1;

  // Get status color for a metric
  const getStatusColor = (deviationPercent: number, invertGood?: boolean) => {
    const isBadDeviation = invertGood ? deviationPercent < -20 : deviationPercent > 20;
    const isGoodDeviation = invertGood ? deviationPercent > 20 : deviationPercent < -20;

    if (isBadDeviation) return '#ef4444'; // danger
    if (isGoodDeviation) return '#10b981'; // success
    if (Math.abs(deviationPercent) > 10) return '#f59e0b'; // warning
    return '#38bdf8'; // accent
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload as ChartDataPoint;
      const color = getStatusColor(data.deviationPercent, data.invertGood);
      return (
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem',
          boxShadow: 'var(--shadow-lg)'
        }}>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>{data.name}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <div>Baseline: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{data.baseline.toFixed(1)}</span></div>
            <div>Current: <span style={{ color, fontWeight: 600 }}>{data.current.toFixed(1)}</span></div>
            <div style={{ color, fontWeight: 600, fontFamily: 'monospace' }}>
              {data.deviation > 0 ? '+' : ''}{data.deviationPercent.toFixed(1)}%
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom dot component
  const CustomDot = ({ cx, cy, payload, ...props }: any) => {
    const color = getStatusColor(payload.deviationPercent, payload.invertGood);
    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={6}
          fill={color}
          stroke="#fff"
          strokeWidth={2}
          style={{ transition: 'all 0.3s ease' }}
        />
        <circle
          cx={cx}
          cy={cy}
          r={3}
          fill="#fff"
          opacity={0.5}
        />
      </g>
    );
  };

  // Custom active dot
  const CustomActiveDot = ({ cx, cy, payload, ...props }: any) => {
    const color = getStatusColor(payload.deviationPercent, payload.invertGood);
    return (
      <g>
        <circle cx={cx} cy={cy} r={10} fill={color} opacity={0.2} />
        <circle
          cx={cx}
          cy={cy}
          r={8}
          fill={color}
          stroke="#fff"
          strokeWidth={3}
        />
      </g>
    );
  };

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '280px' }}>
      <ResponsiveContainer>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="name"
            stroke="var(--text-muted)"
            fontSize={11}
            angle={-25}
            textAnchor="end"
            height={60}
            interval={0}
            tick={{ fill: 'var(--text-secondary)' }}
          />
          <YAxis
            stroke="var(--text-muted)"
            fontSize={11}
            domain={[minValue, maxValue]}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Reference line at zero deviation (optional visual guide) */}
          <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="3 3" />

          {/* Baseline line - subtle reference */}
          <Line
            type="monotone"
            dataKey="baseline"
            stroke="#64748b"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            activeDot={false}
            name="Baseline"
          />

          {/* Current values line - primary visualization */}
          <Line
            type="monotone"
            dataKey="current"
            stroke="#38bdf8"
            strokeWidth={3}
            dot={(props: any) => <CustomDot {...props} />}
            activeDot={(props: any) => <CustomActiveDot {...props} />}
            name="Current"
          />

          {/* Deviation area fill between baseline and current */}
          {/* Note: Recharts doesn't have built-in area between lines, using gradient instead */}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Baseline deviation chart — plots % deviation from baseline for all features
const BaselineSlopeChart: React.FC<{ metrics: BaselineMetric[] }> = ({ metrics }) => {
  // Transform metrics into % deviation from baseline
  const chartData = metrics.map((metric, index) => {
    // Division-by-zero guard: baseline is pre-filtered to !== 0 in PatientDetail,
    // but double-guard here as well
    const deviation = metric.baseline !== 0
      ? ((metric.current - metric.baseline) / metric.baseline) * 100
      : 0;
    return {
      name: metric.label,
      deviation,
      rawCurrent: metric.current,
      rawBaseline: metric.baseline,
      unit: metric.unit,
      order: index,
      invertGood: metric.invertGood,
    };
  });

  // Symmetric Y-axis domain centered on 0
  const allDeviations = chartData.map(d => d.deviation);
  const absMax = Math.max(
    Math.abs(Math.min(...allDeviations)),
    Math.abs(Math.max(...allDeviations)),
    10 // minimum ±10% range
  );
  const domainBound = Math.ceil(absMax * 1.2);

  // Deviation severity color helper
  const getDevColor = (dev: number, invertGood?: boolean) => {
    const absDev = Math.abs(dev);
    if (absDev <= 10) return '#10b981'; // stable — green
    if (absDev <= 20) return '#f59e0b'; // warning — amber
    // Large deviations — check direction relative to invertGood
    const isHigh = dev > 0;
    const isBad = invertGood ? isHigh : !isHigh;
    return isBad ? '#ef4444' : '#10b981';
  };

  // Custom tooltip — shows BOTH raw value (for clinicians) AND % deviation
  const CustomDeviationTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload;
      const dev = data.deviation as number;
      const color = getDevColor(dev, data.invertGood);
      return (
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem',
          boxShadow: 'var(--shadow-lg)',
          minWidth: '170px',
        }}>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
            {data.name}
          </div>
          <div style={{ fontSize: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <div style={{ color: 'var(--text-secondary)' }}>
              Current: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{data.rawCurrent.toFixed(1)} {data.unit}</span>
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>
              Baseline: <span style={{ color: '#64748b', fontWeight: 600 }}>{data.rawBaseline.toFixed(1)} {data.unit}</span>
            </div>
            <div style={{
              color, fontWeight: 700, fontFamily: 'monospace',
              marginTop: '0.25rem', fontSize: '0.9rem',
              padding: '0.2rem 0.5rem',
              background: `${color}12`,
              borderRadius: '4px',
              display: 'inline-block',
              width: 'fit-content'
            }}>
              {dev > 0 ? '+' : ''}{dev.toFixed(1)}% from baseline
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom dot — color-coded by deviation severity
  const DeviationDot = ({ cx, cy, payload }: any) => {
    if (cx === undefined || cy === undefined) return null;
    const color = getDevColor(payload.deviation, payload.invertGood);
    return (
      <g>
        <circle cx={cx} cy={cy} r={5} fill={color} stroke="#fff" strokeWidth={2}
          style={{ transition: 'all 0.3s ease' }} />
      </g>
    );
  };

  const DeviationActiveDot = ({ cx, cy, payload }: any) => {
    if (cx === undefined || cy === undefined) return null;
    const color = getDevColor(payload.deviation, payload.invertGood);
    return (
      <g>
        <circle cx={cx} cy={cy} r={10} fill={color} opacity={0.2} />
        <circle cx={cx} cy={cy} r={7} fill={color} stroke="#fff" strokeWidth={2} />
      </g>
    );
  };

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '320px' }}>
      <ResponsiveContainer>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 30, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="name"
            stroke="var(--text-muted)"
            fontSize={10}
            angle={-45}
            textAnchor="end"
            height={90}
            interval={0}
            tick={{ fill: 'var(--text-secondary)' }}
          />
          <YAxis
            stroke="var(--text-muted)"
            fontSize={11}
            domain={[-domainBound, domainBound]}
            tickFormatter={(v: number) => `${v > 0 ? '+' : ''}${v.toFixed(0)}%`}
          />
          <Tooltip content={<CustomDeviationTooltip />} />

          {/* 0% reference line = "at personal baseline" */}
          <ReferenceLine
            y={0}
            stroke="#64748b"
            strokeWidth={2}
            strokeDasharray="4 4"
            label={{ position: 'insideTopRight', value: 'Baseline (0%)', fill: '#64748b', fontSize: 10, fontWeight: 600 }}
          />

          {/* % deviation line — all features on a single normalized axis */}
          <Line
            type="monotone"
            dataKey="deviation"
            stroke="#38bdf8"
            strokeWidth={3}
            dot={(props: any) => <DeviationDot {...props} />}
            activeDot={(props: any) => <DeviationActiveDot {...props} />}
            name="% Deviation"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export { BaselineLineGraph, BaselineSlopeChart };
