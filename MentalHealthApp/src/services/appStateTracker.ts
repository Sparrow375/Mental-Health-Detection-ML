/**
 * AppStateTracker
 * Singleton that monitors AppState changes to derive:
 *   - screen_time_hours   (time app/screen is active)
 *   - unlock_count        (background→active transitions)
 *   - dark_duration_hours (time screen is off / in background)
 *   - wake_time_hour      (first active time of day)
 *   - sleep_time_hour     (last active time before long idle)
 */
import { AppState, AppStateStatus } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeUsageStats } from './nativeUsageStats';

const KEY_PREFIX = '@mhd_ast_';

function todayKey(suffix: string) {
    const d = new Date();
    const ymd = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    return `${KEY_PREFIX}${ymd}_${suffix}`;
}

async function getNum(key: string, def = 0): Promise<number> {
    try { const v = await AsyncStorage.getItem(key); return v != null ? parseFloat(v) : def; } catch { return def; }
}
async function setNum(key: string, v: number) {
    try { await AsyncStorage.setItem(key, String(v)); } catch { }
}

let _started = false;
let _lastActiveSince: number | null = null;
let _subscription: ReturnType<typeof AppState.addEventListener> | null = null;

export function startAppStateTracker() {
    if (_started) return;
    _started = true;

    const onStateChange = async (next: AppStateStatus) => {
        const now = Date.now();
        if (next === 'active') {
            _lastActiveSince = now;
            // count unlock
            const cnt = await getNum(todayKey('unlock_count'), 0);
            await setNum(todayKey('unlock_count'), cnt + 1);
            // record wake time (first active of day)
            const wakeSet = await getNum(todayKey('wake_set'), 0);
            if (!wakeSet) {
                const hour = new Date().getHours() + new Date().getMinutes() / 60;
                await setNum(todayKey('wake_time'), hour);
                await setNum(todayKey('wake_set'), 1);
            }
        } else if (next === 'background' || next === 'inactive') {
            if (_lastActiveSince != null) {
                const dur = (now - _lastActiveSince) / 3600000; // hours
                const screenTime = await getNum(todayKey('screen_time'), 0);
                await setNum(todayKey('screen_time'), screenTime + dur);
                // track last active as potential sleep time
                const hour = new Date().getHours() + new Date().getMinutes() / 60;
                await setNum(todayKey('last_active_hour'), hour);
                _lastActiveSince = null;
            }
            // accumulate dark time
            const darkStart = await getNum(todayKey('dark_start'), 0);
            if (!darkStart) await setNum(todayKey('dark_start'), now);
        }
    };

    // initialise based on current state
    if (AppState.currentState === 'active') {
        _lastActiveSince = Date.now();
    }

    _subscription = AppState.addEventListener('change', onStateChange);
}

export function stopAppStateTracker() {
    _subscription?.remove();
    _subscription = null;
    _started = false;
}

/** Read today's accumulated AppState metrics */
export async function getTodayAppStateMetrics() {
    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);
    const startMs = startOfDay.getTime();
    const endMs = Date.now();

    const hasNativePerm = await NativeUsageStats.checkPermission();
    let nativeScreenTime = -1;
    let nativeUnlockCount = -1;

    if (hasNativePerm) {
        try {
            nativeScreenTime = await NativeUsageStats.getTotalScreenTimeHours(startMs, endMs);
            nativeUnlockCount = await NativeUsageStats.getUnlockCount(startMs, endMs);
        } catch (e) {
            console.warn('[AppStateTracker] Native stats fetch failed:', e);
        }
    }

    const [screenTime, unlockCount, wakeTime, lastActiveHour, darkStart] = await Promise.all([
        getNum(todayKey('screen_time'), 0),
        getNum(todayKey('unlock_count'), 0),
        getNum(todayKey('wake_time'), 7.5),
        getNum(todayKey('last_active_hour'), 23.0),
        getNum(todayKey('dark_start'), 0),
    ]);

    // use native if valid (> -1), otherwise use manual
    const finalScreenTime = nativeScreenTime >= 0 ? nativeScreenTime : screenTime;
    const finalUnlockCount = nativeUnlockCount >= 0 ? nativeUnlockCount : unlockCount;

    // add ongoing active session if still active AND not using native (native usually updates every few minutes)
    let totalScreen = finalScreenTime;
    if (nativeScreenTime < 0 && _lastActiveSince != null) {
        totalScreen += (Date.now() - _lastActiveSince) / 3600000;
    }

    // dark duration logic remains similar but could also be derived from native events
    let darkDuration = 0;
    if (darkStart > 0 && _lastActiveSince == null) {
        darkDuration = (Date.now() - darkStart) / 3600000;
    }
    const savedDark = await getNum(todayKey('dark_duration'), 0);
    darkDuration += savedDark;

    return {
        screen_time_hours: Math.min(totalScreen, 18),
        unlock_count: Math.min(finalUnlockCount, 300),
        wake_time_hour: wakeTime,
        sleep_time_hour: lastActiveHour,
        dark_duration_hours: Math.min(darkDuration, 24),
    };
}
