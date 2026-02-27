import {
    AlertLevel, AnomalyReport, DailyReport, DetectorState,
    FEATURE_KEYS, FinalPrediction, PatternType, PersonalityVector,
} from './types';

const HISTORY_WINDOW = 7;
const ANOMALY_SCORE_HISTORY_MAX = 14;
const ANOMALY_ALPHA = 0.4;
const SUSTAINED_THRESHOLD_DAYS = 4;
const EVIDENCE_THRESHOLD = 2.0;
const ANOMALY_SCORE_THRESHOLD = 0.35;
const PEAK_EVIDENCE_THRESHOLD = 2.7;
const PEAK_SUSTAINED_THRESHOLD_DAYS = 5;

function mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function std(arr: number[]): number {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}
function polyfit1(x: number[], y: number[]): number {
    const n = x.length; const mx = mean(x); const my = mean(y);
    const num = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0);
    const den = x.reduce((s, xi) => s + (xi - mx) ** 2, 0);
    return den === 0 ? 0 : num / den;
}

export function createDefaultState(baseline: PersonalityVector): DetectorState {
    const featureHistory: Record<string, number[]> = {};
    for (const k of FEATURE_KEYS) featureHistory[k] = [];
    return {
        sustainedDeviationDays: 0, evidenceAccumulated: 0, maxEvidence: 0,
        maxSustainedDays: 0, maxAnomalyScore: 0, anomalyScoreHistory: [], featureHistory
    };
}

export function calculateDeviationMagnitude(
    current: Partial<Record<string, number>>, baseline: PersonalityVector
): Record<string, number> {
    const deviations: Record<string, number> = {};
    for (const feat of FEATURE_KEYS) {
        const baseVal = (baseline as any)[feat] as number;
        const curVal = (current[feat] ?? baseVal) as number;
        const variance = baseline.variances[feat] ?? 0;
        deviations[feat] = variance > 0 ? (curVal - baseVal) / variance : 0;
    }
    return deviations;
}

export function calculateDeviationVelocity(
    current: Partial<Record<string, number>>, baseline: PersonalityVector, state: DetectorState
): { velocities: Record<string, number>; updatedHistories: Record<string, number[]> } {
    const velocities: Record<string, number> = {};
    const updatedHistories: Record<string, number[]> = {};
    for (const feat of FEATURE_KEYS) {
        const val = (current[feat] ?? (baseline as any)[feat]) as number;
        const history = [...(state.featureHistory[feat] ?? []), val].slice(-HISTORY_WINDOW);
        updatedHistories[feat] = history;
        if (history.length < 2) { velocities[feat] = 0; continue; }
        let ewma = history[0];
        const ewmaValues: number[] = [ewma];
        for (let i = 1; i < history.length; i++) {
            ewma = ANOMALY_ALPHA * history[i] + (1 - ANOMALY_ALPHA) * ewma;
            ewmaValues.push(ewma);
        }
        const slope = (ewmaValues[ewmaValues.length - 1] - ewmaValues[0]) / ewmaValues.length;
        const baseVal = (baseline as any)[feat] as number;
        velocities[feat] = baseVal > 0 ? slope / baseVal : 0;
    }
    return { velocities, updatedHistories };
}

export function detectPatternType(deviationsHistory: Record<string, number>[]): PatternType {
    if (deviationsHistory.length < 7) return 'insufficient_data';
    const recent = deviationsHistory.slice(-7);
    const avgDeviations = recent.map(d => mean(Object.values(d).map(Math.abs)));
    const meanDev = mean(avgDeviations); const stdDev = std(avgDeviations);
    if (meanDev < 0.5) return 'stable';
    if (stdDev > 1.0 && meanDev > 0.5) return 'rapid_cycling';
    if (meanDev > 1.5 && stdDev < 0.8) return 'acute_spike';
    const slope = polyfit1(avgDeviations.map((_, i) => i), avgDeviations);
    return Math.abs(slope) > 0.1 ? 'gradual_drift' : 'mixed_pattern';
}

export function calculateAnomalyScore(
    deviations: Record<string, number>, velocities: Record<string, number>
): number {
    const m = Math.min(mean(Object.values(deviations).map(Math.abs)) / 3.0, 1);
    const v = Math.min(mean(Object.values(velocities).map(Math.abs)) * 10, 1);
    return 0.7 * m + 0.3 * v;
}

export function updateSustainedTracking(anomalyScore: number, state: DetectorState): DetectorState {
    const history = [...state.anomalyScoreHistory, anomalyScore].slice(-ANOMALY_SCORE_HISTORY_MAX);
    let { sustainedDeviationDays: sdd, evidenceAccumulated: ea, maxEvidence: me,
        maxSustainedDays: msd, maxAnomalyScore: mas } = state;
    mas = Math.max(mas, anomalyScore);
    if (anomalyScore > ANOMALY_SCORE_THRESHOLD) {
        sdd += 1; ea += anomalyScore * (1 + sdd * 0.1);
    } else { sdd = Math.max(0, sdd - 1); ea *= 0.92; }
    me = Math.max(me, ea); msd = Math.max(msd, sdd);
    return {
        ...state, anomalyScoreHistory: history,
        sustainedDeviationDays: sdd, evidenceAccumulated: ea,
        maxEvidence: me, maxSustainedDays: msd, maxAnomalyScore: mas
    };
}

export function determineAlertLevel(
    anomalyScore: number, deviations: Record<string, number>, state: DetectorState
): AlertLevel {
    const critDev = Math.max(...['screen_time_hours', 'sleep_duration_hours', 'daily_displacement_km']
        .map(f => Math.abs(deviations[f] ?? 0)));
    const hasSustained = state.sustainedDeviationDays >= SUSTAINED_THRESHOLD_DAYS
        || state.evidenceAccumulated >= EVIDENCE_THRESHOLD;
    if (!hasSustained) return 'green';
    if (anomalyScore < 0.35 && critDev < 2.0) return 'green';
    if (anomalyScore < 0.50 && critDev < 2.5) return 'yellow';
    if (anomalyScore < 0.65 || critDev < 3.0) return 'orange';
    return 'red';
}

export function identifyFlaggedFeatures(deviations: Record<string, number>, threshold = 1.5): string[] {
    return Object.entries(deviations).filter(([, v]) => Math.abs(v) > threshold)
        .map(([k, v]) => `${k} (${v.toFixed(2)} SD)`);
}

export function getTopDeviations(deviations: Record<string, number>, topN = 5): Record<string, number> {
    return Object.fromEntries(Object.entries(deviations)
        .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a)).slice(0, topN));
}

function generateNotes(
    anomalyScore: number, alertLevel: AlertLevel, patternType: PatternType, state: DetectorState
): string {
    const parts: string[] = [];
    if (state.sustainedDeviationDays >= SUSTAINED_THRESHOLD_DAYS)
        parts.push(`Sustained deviation (${state.sustainedDeviationDays} consecutive days)`);
    if (state.evidenceAccumulated >= EVIDENCE_THRESHOLD)
        parts.push(`Evidence accumulated: ${state.evidenceAccumulated.toFixed(2)}`);
    if (patternType === 'rapid_cycling' || patternType === 'gradual_drift')
        parts.push(`Pattern: ${patternType}`);
    if (alertLevel === 'orange' || alertLevel === 'red')
        parts.push(`HIGH ALERT: ${alertLevel.toUpperCase()}`);
    if (anomalyScore > 0.6 && alertLevel === 'green')
        parts.push('High single-day score but no sustained pattern yet');
    return parts.length > 0 ? parts.join(' | ') : 'Normal operation';
}

export interface AnalyzeResult {
    anomalyReport: AnomalyReport; dailyReport: DailyReport; newState: DetectorState;
}

export function analyze(
    current: Partial<Record<string, number>>, baseline: PersonalityVector,
    state: DetectorState, deviationsHistory: Record<string, number>[],
    dayNumber: number, featuresEntered: Partial<Omit<PersonalityVector, 'variances'>>
): AnalyzeResult {
    const deviations = calculateDeviationMagnitude(current, baseline);
    const { velocities, updatedHistories } = calculateDeviationVelocity(current, baseline, state);
    const anomalyScore = calculateAnomalyScore(deviations, velocities);
    const stateAfter = updateSustainedTracking(anomalyScore, { ...state, featureHistory: updatedHistories });
    const alertLevel = determineAlertLevel(anomalyScore, deviations, stateAfter);
    const flaggedFeatures = identifyFlaggedFeatures(deviations);
    const patternType = detectPatternType(deviationsHistory);
    const topDeviations = getTopDeviations(deviations);
    const notes = generateNotes(anomalyScore, alertLevel, patternType, stateAfter);
    const now = new Date().toISOString();

    return {
        anomalyReport: {
            timestamp: now, overallAnomalyScore: anomalyScore,
            featureDeviations: deviations, deviationVelocity: velocities, alertLevel,
            flaggedFeatures, patternType, sustainedDeviationDays: stateAfter.sustainedDeviationDays,
            evidenceAccumulated: stateAfter.evidenceAccumulated
        },
        dailyReport: {
            dayNumber, date: now, anomalyScore, alertLevel, flaggedFeatures,
            patternType, sustainedDeviationDays: stateAfter.sustainedDeviationDays,
            evidenceAccumulated: stateAfter.evidenceAccumulated, topDeviations, notes,
            featuresEntered
        },
        newState: stateAfter,
    };
}

export function generateFinalPrediction(
    patientId: string, monitoringDays: number, state: DetectorState
): FinalPrediction {
    const confidence = Math.min(0.95, (monitoringDays / 30) * 0.8 + 0.15);
    const sustainedAnomaly = state.maxEvidence >= PEAK_EVIDENCE_THRESHOLD
        || state.maxSustainedDays >= PEAK_SUSTAINED_THRESHOLD_DAYS;
    const finalScore = state.anomalyScoreHistory.length > 0 ? mean(state.anomalyScoreHistory) : 0;
    let pattern = 'stable';
    if (state.anomalyScoreHistory.length >= 7) {
        const recent = state.anomalyScoreHistory.slice(-7);
        if (std(recent) > 0.15) pattern = 'unstable/cycling';
        else if (mean(recent) > 0.5) pattern = 'persistent_elevation';
    }
    const recommendation = sustainedAnomaly && state.maxEvidence >= 4.0
        ? 'REFER: Very strong evidence of sustained behavioral deviation. Immediate clinical evaluation recommended.'
        : sustainedAnomaly
            ? 'MONITOR: Significant sustained deviation detected. Clinical follow-up recommended.'
            : state.maxEvidence > 1.5
                ? 'WATCH: Some periodic evidence of deviation. Suggest extending monitoring.'
                : 'NORMAL: No significant sustained deviation detected during the study period.';
    return {
        patientId, monitoringDays, finalAnomalyScore: finalScore,
        sustainedAnomalyDetected: sustainedAnomaly, confidence,
        patternIdentified: pattern,
        evidenceSummary: {
            sustainedDeviationDays: state.sustainedDeviationDays,
            maxSustainedDays: state.maxSustainedDays,
            evidenceAccumulatedFinal: Math.round(state.evidenceAccumulated * 100) / 100,
            peakEvidence: Math.round(state.maxEvidence * 100) / 100,
            maxDailyAnomalyScore: Math.round(state.maxAnomalyScore * 1000) / 1000,
            avgRecentAnomalyScore: Math.round(finalScore * 1000) / 1000,
            monitoringDays,
            daysAboveThreshold: state.anomalyScoreHistory.filter(s => s > ANOMALY_SCORE_THRESHOLD).length,
        },
        recommendation,
    };
}
