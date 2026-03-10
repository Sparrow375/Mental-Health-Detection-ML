import { NativeModules, Platform } from 'react-native';

const { UsageStatsModule } = NativeModules;

export interface AppUsageStats {
    packageName: string;
    totalTimeInForeground: number; // ms
    lastTimeUsed: number; // ts
}

export interface ScreenEvent {
    type: number; // 15=SCREEN_INTERACTIVE, 16=SCREEN_NON_INTERACTIVE
    timestamp: number;
    packageName: string;
}

const NATIVE_ENABLED = Platform.OS === 'android' && UsageStatsModule;

export const NativeUsageStats = {
    async checkPermission(): Promise<boolean> {
        if (!NATIVE_ENABLED) return false;
        return UsageStatsModule.checkPermission();
    },

    async requestPermission() {
        if (!NATIVE_ENABLED) return;
        UsageStatsModule.showUsageAccessSettings();
    },

    async getDailyStats(startTime: number, endTime: number): Promise<AppUsageStats[]> {
        if (!NATIVE_ENABLED) return [];
        return UsageStatsModule.getDailyStats(startTime, endTime);
    },

    async getScreenEvents(startTime: number, endTime: number): Promise<ScreenEvent[]> {
        if (!NATIVE_ENABLED) return [];
        return UsageStatsModule.getScreenEvents(startTime, endTime);
    },

    /**
     * Helper to compute total screen time in hours for the period
     */
    async getTotalScreenTimeHours(startTime: number, endTime: number): Promise<number> {
        const stats = await this.getDailyStats(startTime, endTime);
        const totalMs = stats.reduce((acc, curr) => acc + curr.totalTimeInForeground, 0);
        return totalMs / 3600000;
    },

    /**
     * Helper to compute unlock count based on screen events
     */
    async getUnlockCount(startTime: number, endTime: number): Promise<number> {
        const events = await this.getScreenEvents(startTime, endTime);
        // type 15 is SCREEN_INTERACTIVE (usually means turned on)
        // type 26 is KEYGUARD_INTERACTIVE (lock screen shown)
        // type 27 is KEYGUARD_GONE (unlocked)
        // In many Android versions, 15 + 27 together indicate a full unlock cycle.
        // We'll count 15 (Screen On) or 27 (Keyguard Gone) as our unlock markers.
        const unlocks = events.filter(e => e.type === 15 || e.type === 27);
        return unlocks.length;
    }
};
