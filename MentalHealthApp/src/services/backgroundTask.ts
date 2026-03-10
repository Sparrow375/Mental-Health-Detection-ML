/**
 * Background Task Registration
 * Uses expo-task-manager + expo-background-fetch to:
 *  1. Capture location snapshots every ~15 minutes
 *  2. Run end-of-day analysis at midnight / early morning
 */
import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';
import * as Location from 'expo-location';
import { recordBatterySnapshot, storeLocationSnapshot, collectTodayData, saveTodaySnapshot } from './autoCollector';
import { getBaseline, getDailyReports, getDetectorState, appendDailyReport, saveDetectorState, appendDeviationsHistory } from './storage';
import { analyze } from '../engine/anomalyDetector';
import { sendAlertNotification } from './notifications';

export const BG_LOCATION_TASK = 'mhd-bg-location';
export const BG_FETCH_TASK = 'mhd-bg-fetch';

// ── Background Location Task ──────────────────────────────────────────────────
TaskManager.defineTask(BG_LOCATION_TASK, async ({ data, error }: any) => {
    if (error) { console.warn('[BG_LOC] error:', error); return; }
    if (data?.locations) {
        await storeLocationSnapshot();
        await recordBatterySnapshot();
    }
});

// ── Background Fetch Task ─────────────────────────────────────────────────────
TaskManager.defineTask(BG_FETCH_TASK, async () => {
    try {
        await storeLocationSnapshot();
        await recordBatterySnapshot();

        // Check if we should run daily analysis (once per day, after midnight)
        const now = new Date();
        const hour = now.getHours();
        if (hour >= 0 && hour <= 4) {
            // Run auto-analysis for yesterday's data
            await runDailyAnalysis();
        }

        return BackgroundFetch.BackgroundFetchResult.NewData;
    } catch {
        return BackgroundFetch.BackgroundFetchResult.Failed;
    }
});

// ── Daily Auto-Analysis ───────────────────────────────────────────────────────
export async function runDailyAnalysis(): Promise<boolean> {
    try {
        const [collected, baseline, existing] = await Promise.all([
            collectTodayData(),
            getBaseline(),
            getDailyReports(),
        ]);
        await saveTodaySnapshot(collected);
        const state = await getDetectorState(baseline);
        const deviationsHistory = existing.map(r => r.topDeviations);
        const { anomalyReport, dailyReport, newState } = analyze(
            collected, baseline, state, deviationsHistory, existing.length + 1, collected as any
        );
        await Promise.all([
            appendDailyReport(dailyReport),
            saveDetectorState(newState),
            appendDeviationsHistory(anomalyReport.featureDeviations),
        ]);
        await sendAlertNotification(dailyReport.alertLevel, dailyReport.anomalyScore, dailyReport.flaggedFeatures);
        return true;
    } catch (e) {
        console.warn('[runDailyAnalysis] failed:', e);
        return false;
    }
}

// ── Registration ──────────────────────────────────────────────────────────────
export async function registerBackgroundTasks() {
    // Background fetch — fires every 15 minutes (minimum allowed)
    try {
        await BackgroundFetch.registerTaskAsync(BG_FETCH_TASK, {
            minimumInterval: 15 * 60, // 15 min
            stopOnTerminate: false,
            startOnBoot: true,
        });
    } catch (e) {
        console.warn('[registerBackgroundTasks] fetch register failed:', e);
    }

    // Background location — continuous low-accuracy updates
    try {
        const { status } = await Location.requestBackgroundPermissionsAsync();
        if (status === 'granted') {
            const alreadyRunning = await Location.hasStartedLocationUpdatesAsync(BG_LOCATION_TASK);
            if (!alreadyRunning) {
                await Location.startLocationUpdatesAsync(BG_LOCATION_TASK, {
                    accuracy: Location.Accuracy.Balanced,
                    timeInterval: 30 * 60 * 1000,  // every 30 min
                    distanceInterval: 500,           // or every 500m
                    showsBackgroundLocationIndicator: false,
                    foregroundService: {
                        notificationTitle: 'MH Monitor',
                        notificationBody: 'Monitoring your daily patterns in the background.',
                        notificationColor: '#6C63FF',
                    },
                });
            }
        }
    } catch (e) {
        console.warn('[registerBackgroundTasks] location register failed:', e);
    }
}

export async function unregisterBackgroundTasks() {
    try {
        const isFetchRegistered = await TaskManager.isTaskRegisteredAsync(BG_FETCH_TASK);
        if (isFetchRegistered) await BackgroundFetch.unregisterTaskAsync(BG_FETCH_TASK);
    } catch { }
    try {
        const isLocRunning = await Location.hasStartedLocationUpdatesAsync(BG_LOCATION_TASK);
        if (isLocRunning) await Location.stopLocationUpdatesAsync(BG_LOCATION_TASK);
    } catch { }
}
