import { NativeUsageStats } from './nativeUsageStats';
import { FEATURE_KEYS, FEATURE_DEFAULTS, PersonalityVector } from '../engine/types';
import { saveBaseline } from './storage';

/**
 * Extracts data for the last few days to build a real baseline.
 * If data is missing for some days/metrics, it falls back to defaults.
 */
export async function buildHistoricalBaseline(days: number = 7): Promise<PersonalityVector> {
    const hasPerm = await NativeUsageStats.checkPermission();
    if (!hasPerm) {
        console.warn('[DataExtractor] No native permission, cannot build historical baseline.');
        return generateDefaultBaseline();
    }

    const accum: Record<string, number[]> = {};
    for (const k of FEATURE_KEYS) accum[k] = [];

    const now = Date.now();
    for (let i = 1; i <= days; i++) {
        const start = now - (i * 24 * 3600000);
        const end = start + (24 * 3600000);

        // Get daily metrics
        const screenTime = await NativeUsageStats.getTotalScreenTimeHours(start, end);
        const unlockCount = await NativeUsageStats.getUnlockCount(start, end);

        // For now we only have these two from native. The rest were synthetic or manual.
        // We can append more native stats later (location historical? probably not easily without a DB).

        accum['screen_time_hours'].push(screenTime);
        accum['unlock_count'].push(unlockCount);

        // Fill other features with defaults for now, or random variance around defaults
        for (const k of FEATURE_KEYS) {
            if (k !== 'screen_time_hours' && k !== 'unlock_count') {
                const def = (FEATURE_DEFAULTS as any)[k] ?? 0;
                accum[k].push(def + (Math.random() - 0.5) * def * 0.1);
            }
        }
    }

    const means: Record<string, number> = {};
    const variances: Record<string, number> = {};

    for (const k of FEATURE_KEYS) {
        const vals = accum[k];
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        const s = Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / vals.length);
        means[k] = m;
        variances[k] = Math.max(s, 0.001);
    }

    const baseline = { ...(means as any), variances } as PersonalityVector;
    await saveBaseline(baseline);
    return baseline;
}

function generateDefaultBaseline(): PersonalityVector {
    const result: Record<string, number> = {};
    const variances: Record<string, number> = {};
    for (const k of FEATURE_KEYS) {
        const val = (FEATURE_DEFAULTS as any)[k];
        result[k] = val;
        variances[k] = Math.max(val * 0.12, 0.001);
    }
    return { ...(result as any), variances } as PersonalityVector;
}
