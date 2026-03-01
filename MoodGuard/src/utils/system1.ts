// src/utils/system1.ts
export type FeatureKey =
  | 'screen_time_hours'
  | 'sleep_duration_hours'
  | 'daily_displacement_km'
  | 'texts_per_day'
  | 'voice_energy_mean';

export type DailyInput = {
  date: string; // ISO string YYYY-MM-DD
  screen_time_hours: number;
  sleep_duration_hours: number;
  daily_displacement_km: number;
  texts_per_day: number;
  voice_energy_mean: number;
};

export type BaselineStats = {
  mean: Record<FeatureKey, number>;
  std: Record<FeatureKey, number>;
};

export type DailyAnalysis = {
  date: string;
  anomalyScore: number; // 0-1
  alertLevel: 'green' | 'yellow' | 'orange' | 'red';
  sustainedDeviationDays: number;
  evidenceAccumulated: number;
};

export type AnalysisSummary = {
  baseline: BaselineStats | null;
  history: DailyAnalysis[];
  latest: DailyAnalysis | null;
};

const FEATURES: FeatureKey[] = [
  'screen_time_hours',
  'sleep_duration_hours',
  'daily_displacement_km',
  'texts_per_day',
  'voice_energy_mean',
];

// Tunable thresholds (mirroring spirit of Python values)
const ANOMALY_SCORE_THRESHOLD = 0.35;
const SUSTAINED_THRESHOLD_DAYS = 4;
const EVIDENCE_THRESHOLD = 2.0;

// --- Helpers ---

function computeMean(values: number[]): number {
  if (!values.length) return 0;
  const sum = values.reduce((a, b) => a + b, 0);
  return sum / values.length;
}

function computeStd(values: number[], mean: number): number {
  if (values.length < 2) return 0.0001;
  const variance =
    values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) /
    (values.length - 1);
  return Math.sqrt(variance) || 0.0001;
}

// --- Baseline computation (first N days become "healthy baseline") ---

export function computeBaseline(
  data: DailyInput[],
  baselineDays: number = 14,
): BaselineStats | null {
  if (data.length < baselineDays) return null;

  const slice = data.slice(0, baselineDays);
  const mean: Record<FeatureKey, number> = {} as Record<FeatureKey, number>;
  const std: Record<FeatureKey, number> = {} as Record<FeatureKey, number>;

  FEATURES.forEach((f) => {
    const vals = slice.map((d) => d[f]);
    const m = computeMean(vals);
    mean[f] = m;
    std[f] = computeStd(vals, m);
  });

  return { mean, std };
}

// --- Core anomaly computation over full history ---

export function analyzeHistory(inputs: DailyInput[]): AnalysisSummary {
  if (!inputs.length) {
    return { baseline: null, history: [], latest: null };
  }

  const baseline = computeBaseline(inputs);
  if (!baseline) {
    // Not enough data yet to build baseline – always green
    const history: DailyAnalysis[] = inputs.map((d) => ({
      date: d.date,
      anomalyScore: 0,
      alertLevel: 'green' as const,
      sustainedDeviationDays: 0,
      evidenceAccumulated: 0,
    }));
    return { baseline: null, history, latest: history[history.length - 1] };
  }

  const history: DailyAnalysis[] = [];
  let sustainedDeviationDays = 0;
  let evidenceAccumulated = 0;

  inputs.forEach((entry) => {
    // Z-score deviations
    const deviations: number[] = FEATURES.map((f) => {
      const m = baseline.mean[f];
      const s = baseline.std[f];
      return (entry[f] - m) / s;
    });

    // Overall anomaly score (0–1) – mean |z|, capped at 3 SD
    let magnitudeScore = computeMean(deviations.map((d) => Math.abs(d)));
    magnitudeScore = Math.min(magnitudeScore / 3.0, 1.0);

    const anomalyScore = magnitudeScore;

    // Update sustained tracking
    if (anomalyScore > ANOMALY_SCORE_THRESHOLD) {
      sustainedDeviationDays += 1;
      evidenceAccumulated += anomalyScore * (1 + sustainedDeviationDays * 0.1);
    } else {
      sustainedDeviationDays = Math.max(0, sustainedDeviationDays - 1);
      evidenceAccumulated *= 0.92;
    }

    // Alert level (similar spirit to Python)
    const criticalFeatures: FeatureKey[] = [
      'voice_energy_mean',
      'sleep_duration_hours',
      'screen_time_hours',
      'daily_displacement_km',
    ];
    const criticalDev = Math.max(
      ...criticalFeatures.map((f) => {
        const m = baseline.mean[f];
        const s = baseline.std[f];
        return Math.abs((entry[f] - m) / s);
      }),
    );

    const hasSustained =
      sustainedDeviationDays >= SUSTAINED_THRESHOLD_DAYS ||
      evidenceAccumulated >= EVIDENCE_THRESHOLD;

    let alertLevel: DailyAnalysis['alertLevel'] = 'green';

    if (!hasSustained) {
      alertLevel = 'green';
    } else if (anomalyScore < 0.35 && criticalDev < 2.0) {
      alertLevel = 'green';
    } else if (anomalyScore < 0.5 && criticalDev < 2.5) {
      alertLevel = 'yellow';
    } else if (anomalyScore < 0.65 || criticalDev < 3.0) {
      alertLevel = 'orange';
    } else {
      alertLevel = 'red';
    }

    history.push({
      date: entry.date,
      anomalyScore,
      alertLevel,
      sustainedDeviationDays,
      evidenceAccumulated,
    });
  });

  return {
    baseline,
    history,
    latest: history[history.length - 1],
  };
}
