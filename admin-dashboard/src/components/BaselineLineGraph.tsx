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

// Unified single-line visualization - mixes all metrics into one combined line
const BaselineSlopeChart: React.FC<{ metrics: BaselineMetric[] }> = ({ metrics }) => {
  // Transform metrics into chart data for a single combined line
  const chartData = metrics.map((metric, index) => {
    return {
      name: metric.label,
      value: metric.current,
      baseline: metric.baseline,
      unit: metric.unit,
      order: index,
    };
  });

  // Calculate Y-axis domain based on all values
  const allValues = chartData.flatMap(d => [d.value, d.baseline]);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = (maxValue - minValue) * 0.1 || 1;

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '280px' }}>
      <ResponsiveContainer>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="name"
            stroke="var(--text-muted)"
            fontSize={11}
            angle={-25}
            textAnchor="end"
            height={70}
            interval={0}
            tick={{ fill: 'var(--text-secondary)' }}
          />
          <YAxis
            stroke="var(--text-muted)"
            fontSize={11}
            domain={[minValue - padding, maxValue + padding]}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--shadow-lg)'
            }}
            itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
            labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}
          />

          {/* Baseline line - subtle dashed reference */}
          <Line
            type="monotone"
            dataKey="baseline"
            stroke="#64748b"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
            activeDot={false}
            name="Baseline"
          />

          {/* Single unified current values line - no deviations, just the data */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#38bdf8"
            strokeWidth={3}
            dot={{ r: 5, fill: '#38bdf8', stroke: '#fff', strokeWidth: 2 }}
            activeDot={{ r: 8, fill: '#38bdf8', stroke: '#fff', strokeWidth: 2 }}
            name="Current"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export { BaselineLineGraph, BaselineSlopeChart };
