/**
 * AutoCollector
 * Gathers all sensor data from the device for a given day and returns
 * a complete feature vector. Combines:
 *   - AppStateTracker data (screen, unlocks, wake/sleep times)
 *   - expo-location (displacement, entropy, home ratio, places)
 *   - expo-sensors Pedometer (not used directly — displacement proxy)
 *   - expo-battery (charge duration)
 *   - expo-contacts (unique contacts count)
 */
import * as Location from 'expo-location';
import * as Battery from 'expo-battery';
import * as Contacts from 'expo-contacts';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NativeUsageStats } from './nativeUsageStats';
import { FEATURE_DEFAULTS } from '../engine/types';
import { getTodayAppStateMetrics } from './appStateTracker';

const LOC_KEY_PREFIX = '@mhd_loc_';
const BATT_KEY_PREFIX = '@mhd_batt_';
const HOME_LOC_KEY = '@mhd_home_location';

function todayKey(suffix: string) {
    const d = new Date();
    const ymd = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    return `${LOC_KEY_PREFIX}${ymd}_${suffix}`;
}

// ── Haversine Distance ────────────────────────────────────────────────────────
function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number) {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Location Cluster (500m grid) ──────────────────────────────────────────────
function clusterKey(lat: number, lon: number) {
    return `${Math.round(lat * 200)}_${Math.round(lon * 200)}`; // ~500m bucket
}

function shannonEntropy(counts: number[]): number {
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    return counts.reduce((e, c) => {
        if (c === 0) return e;
        const p = c / total;
        return e - p * Math.log2(p);
    }, 0);
}

// ── Location Snapshot store ────────────────────────────────────────────────────
export async function storeLocationSnapshot() {
    try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') return;
        const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
        const { latitude, longitude } = pos.coords;
        const snapshots = await getLocationSnapshots();
        snapshots.push({ lat: latitude, lon: longitude, ts: Date.now() });
        await AsyncStorage.setItem(todayKey('snapshots'), JSON.stringify(snapshots));

        // save home location if not set (very first run)
        const homeRaw = await AsyncStorage.getItem(HOME_LOC_KEY);
        if (!homeRaw) {
            await AsyncStorage.setItem(HOME_LOC_KEY, JSON.stringify({ lat: latitude, lon: longitude }));
        }
    } catch { }
}

async function getLocationSnapshots(): Promise<{ lat: number; lon: number; ts: number }[]> {
    try {
        const raw = await AsyncStorage.getItem(todayKey('snapshots'));
        return raw ? JSON.parse(raw) : [];
    } catch { return []; }
}

async function computeLocationFeatures() {
    const snaps = await getLocationSnapshots();
    if (snaps.length < 2) {
        return {
            daily_displacement_km: FEATURE_DEFAULTS.daily_displacement_km,
            location_entropy: FEATURE_DEFAULTS.location_entropy,
            home_time_ratio: FEATURE_DEFAULTS.home_time_ratio,
            places_visited: FEATURE_DEFAULTS.places_visited,
        };
    }

    // Sort by timestamp just in case
    snaps.sort((a, b) => a.ts - b.ts);

    // 1. Cumulative Displacement
    let displacement = 0;
    for (let i = 1; i < snaps.length; i++) {
        displacement += haversineKm(snaps[i - 1].lat, snaps[i - 1].lon, snaps[i].lat, snaps[i].lon);
    }

    // 2. Entropy & Cluster mapping
    const clusterCounts: Record<string, number> = {};
    const nighttimeClusterCounts: Record<string, number> = {};

    for (const s of snaps) {
        const k = clusterKey(s.lat, s.lon);
        clusterCounts[k] = (clusterCounts[k] ?? 0) + 1;

        // Nighttime check (10 PM to 6 AM)
        const hour = new Date(s.ts).getHours();
        if (hour >= 22 || hour < 6) {
            nighttimeClusterCounts[k] = (nighttimeClusterCounts[k] ?? 0) + 1;
        }
    }

    const entropy = shannonEntropy(Object.values(clusterCounts));
    const placesVisited = Object.keys(clusterCounts).length;

    // 3. Identify Home (highest frequency at night)
    let homeCluster: string | null = null;
    let maxNightCount = 0;
    for (const [k, count] of Object.entries(nighttimeClusterCounts)) {
        if (count > maxNightCount) {
            maxNightCount = count;
            homeCluster = k;
        }
    }

    // Fallback: If no nighttime data, use the first location of the day
    if (!homeCluster && snaps.length > 0) {
        homeCluster = clusterKey(snaps[0].lat, snaps[0].lon);
    }

    // Calculate Home Time Ratio (ratio of stay events in the home cluster)
    let homeResidency = 0;
    if (homeCluster) {
        homeResidency = (clusterCounts[homeCluster] || 0) / snaps.length;
    }

    return {
        daily_displacement_km: Math.min(displacement, 100),
        location_entropy: Math.min(entropy, 5),
        home_time_ratio: Math.max(0.1, Math.min(homeResidency, 1.0)),
        places_visited: Math.min(placesVisited, 30),
    };
}

// ── Battery / Charge ──────────────────────────────────────────────────────────
export async function recordBatterySnapshot() {
    try {
        const state = await Battery.getBatteryStateAsync();
        const battKey = `${BATT_KEY_PREFIX}${new Date().toISOString().slice(0, 10)}`;
        const raw = await AsyncStorage.getItem(battKey);
        const history: { state: number; ts: number }[] = raw ? JSON.parse(raw) : [];
        history.push({ state, ts: Date.now() });
        await AsyncStorage.setItem(battKey, JSON.stringify(history));
    } catch { }
}

async function computeChargeHours(): Promise<number> {
    try {
        const battKey = `${BATT_KEY_PREFIX}${new Date().toISOString().slice(0, 10)}`;
        const raw = await AsyncStorage.getItem(battKey);
        if (!raw) return FEATURE_DEFAULTS.charge_duration_hours;
        const history: { state: number; ts: number }[] = JSON.parse(raw);
        let chargeMs = 0;
        let chargingStart: number | null = null;
        for (const h of history) {
            const isCharging = h.state === Battery.BatteryState.CHARGING || h.state === Battery.BatteryState.FULL;
            if (isCharging && chargingStart == null) chargingStart = h.ts;
            else if (!isCharging && chargingStart != null) {
                chargeMs += h.ts - chargingStart;
                chargingStart = null;
            }
        }
        if (chargingStart != null) chargeMs += Date.now() - chargingStart;
        return Math.min(chargeMs / 3600000, 24);
    } catch { return FEATURE_DEFAULTS.charge_duration_hours; }
}

// ── Contacts ──────────────────────────────────────────────────────────────────
async function getUniqueContactsCount(): Promise<number> {
    try {
        const { status } = await Contacts.requestPermissionsAsync();
        if (status !== 'granted') return FEATURE_DEFAULTS.unique_contacts;
        const { data } = await Contacts.getContactsAsync({ fields: [Contacts.Fields.PhoneNumbers] });
        // cap at 100 as per FEATURE_RANGES
        return Math.min(data.length, 100);
    } catch { return FEATURE_DEFAULTS.unique_contacts; }
}

async function computeSocialAppRatio(): Promise<number> {
    const hasPerm = await NativeUsageStats.checkPermission();
    if (!hasPerm) return FEATURE_DEFAULTS.social_app_ratio;

    const start = new Date();
    start.setHours(0, 0, 0, 0);
    const stats = await NativeUsageStats.getDailyStats(start.getTime(), Date.now());

    if (stats.length === 0) return FEATURE_DEFAULTS.social_app_ratio;

    const SOCIAL_PACKAGES = [
        'com.facebook.katana', 'com.instagram.android', 'com.whatsapp',
        'com.twitter.android', 'com.zhiliaoapp.musically', 'com.snapchat.android',
        'com.reddit.frontpage', 'com.linkedin.android', 'org.telegram.messenger'
    ];

    let socialMs = 0;
    let totalMs = 0;

    for (const s of stats) {
        totalMs += s.totalTimeInForeground;
        if (SOCIAL_PACKAGES.includes(s.packageName)) {
            socialMs += s.totalTimeInForeground;
        }
    }

    return totalMs > 0 ? socialMs / totalMs : FEATURE_DEFAULTS.social_app_ratio;
}

// ── Main collection function ───────────────────────────────────────────────────
export async function collectTodayData(): Promise<Record<string, number>> {
    const [appState, locationFeatures, chargeHours, uniqueContacts, socialAppRatio] = await Promise.all([
        getTodayAppStateMetrics(),
        computeLocationFeatures(),
        computeChargeHours(),
        getUniqueContactsCount(),
        computeSocialAppRatio(),
    ]);

    return {
        // From AppStateTracker
        screen_time_hours: appState.screen_time_hours,
        unlock_count: appState.unlock_count,
        wake_time_hour: appState.wake_time_hour,
        sleep_time_hour: appState.sleep_time_hour,
        dark_duration_hours: appState.dark_duration_hours,

        // From location
        daily_displacement_km: locationFeatures.daily_displacement_km,
        location_entropy: locationFeatures.location_entropy,
        home_time_ratio: locationFeatures.home_time_ratio,
        places_visited: locationFeatures.places_visited,

        // From battery
        charge_duration_hours: chargeHours,

        // From contacts
        unique_contacts: uniqueContacts,

        // Sleep duration derived
        sleep_duration_hours: Math.min(
            Math.max(0, ((appState.wake_time_hour + 24) - appState.sleep_time_hour) % 24),
            16
        ),

        // Computed from native stats
        social_app_ratio: socialAppRatio,
        calls_per_day: FEATURE_DEFAULTS.calls_per_day,
        texts_per_day: FEATURE_DEFAULTS.texts_per_day,
        response_time_minutes: FEATURE_DEFAULTS.response_time_minutes,
        conversation_duration_hours: FEATURE_DEFAULTS.conversation_duration_hours,
        conversation_frequency: FEATURE_DEFAULTS.conversation_frequency,
    };
}

/** Save a collected snapshot so it can be displayed on screen */
export async function saveTodaySnapshot(data: Record<string, number>) {
    const key = `@mhd_snapshot_${new Date().toISOString().slice(0, 10)}`;
    await AsyncStorage.setItem(key, JSON.stringify({ data, ts: Date.now() }));
}

export async function getTodaySnapshot(): Promise<{ data: Record<string, number>; ts: number } | null> {
    try {
        const key = `@mhd_snapshot_${new Date().toISOString().slice(0, 10)}`;
        const raw = await AsyncStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
    } catch { return null; }
}

export async function getHomeLocation(): Promise<{ lat: number; lon: number } | null> {
    try {
        const raw = await AsyncStorage.getItem(HOME_LOC_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch { return null; }
}

export async function setHomeLocation(lat: number, lon: number) {
    await AsyncStorage.setItem(HOME_LOC_KEY, JSON.stringify({ lat, lon }));
}
