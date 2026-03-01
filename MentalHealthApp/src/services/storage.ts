import AsyncStorage from '@react-native-async-storage/async-storage';
import { DailyReport, DetectorState, PersonalityVector } from '../engine/types';
import { generateSyntheticBaseline } from '../engine/baselineGenerator';
import { createDefaultState } from '../engine/anomalyDetector';

const KEYS = {
    BASELINE: '@mhd_baseline', REPORTS: '@mhd_daily_reports',
    DETECTOR_STATE: '@mhd_detector_state', PATIENT_ID: '@mhd_patient_id',
    SETUP_DONE: '@mhd_setup_done', NOTIFY_TIME: '@mhd_notify_time',
    DEVIATIONS_HISTORY: '@mhd_deviations_history',
};

export async function getBaseline(): Promise<PersonalityVector> {
    try { const r = await AsyncStorage.getItem(KEYS.BASELINE); if (r) return JSON.parse(r); } catch (_) { }
    return generateSyntheticBaseline();
}
export async function saveBaseline(v: PersonalityVector) {
    await AsyncStorage.setItem(KEYS.BASELINE, JSON.stringify(v));
}

export async function getDailyReports(): Promise<DailyReport[]> {
    try { const r = await AsyncStorage.getItem(KEYS.REPORTS); if (r) return JSON.parse(r); } catch (_) { }
    return [];
}
export async function appendDailyReport(report: DailyReport) {
    const existing = await getDailyReports();
    existing.push(report);
    await AsyncStorage.setItem(KEYS.REPORTS, JSON.stringify(existing));
}
export async function clearAllReports() { await AsyncStorage.removeItem(KEYS.REPORTS); }

export async function getDetectorState(baseline: PersonalityVector): Promise<DetectorState> {
    try { const r = await AsyncStorage.getItem(KEYS.DETECTOR_STATE); if (r) return JSON.parse(r); } catch (_) { }
    return createDefaultState(baseline);
}
export async function saveDetectorState(state: DetectorState) {
    await AsyncStorage.setItem(KEYS.DETECTOR_STATE, JSON.stringify(state));
}

export async function getDeviationsHistory(): Promise<Record<string, number>[]> {
    try { const r = await AsyncStorage.getItem(KEYS.DEVIATIONS_HISTORY); if (r) return JSON.parse(r); } catch (_) { }
    return [];
}
export async function appendDeviationsHistory(dev: Record<string, number>) {
    const existing = await getDeviationsHistory();
    existing.push(dev);
    await AsyncStorage.setItem(KEYS.DEVIATIONS_HISTORY, JSON.stringify(existing.slice(-180)));
}
export async function clearDeviationsHistory() { await AsyncStorage.removeItem(KEYS.DEVIATIONS_HISTORY); }

export async function getPatientId(): Promise<string> {
    try { const v = await AsyncStorage.getItem(KEYS.PATIENT_ID); if (v) return v; } catch (_) { }
    return 'PT-001';
}
export async function savePatientId(id: string) { await AsyncStorage.setItem(KEYS.PATIENT_ID, id); }

export async function isSetupDone(): Promise<boolean> {
    try { return (await AsyncStorage.getItem(KEYS.SETUP_DONE)) === 'true'; } catch (_) { return false; }
}
export async function markSetupDone() { await AsyncStorage.setItem(KEYS.SETUP_DONE, 'true'); }

export async function getNotifyTime(): Promise<string> {
    try { const v = await AsyncStorage.getItem(KEYS.NOTIFY_TIME); if (v) return v; } catch (_) { }
    return '20:00';
}
export async function saveNotifyTime(time: string) { await AsyncStorage.setItem(KEYS.NOTIFY_TIME, time); }

export async function resetAllData() {
    await Promise.all(Object.values(KEYS).map(k => AsyncStorage.removeItem(k)));
}
