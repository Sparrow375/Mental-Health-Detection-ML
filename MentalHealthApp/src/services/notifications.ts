// ============================================================================
// NOTIFICATIONS SERVICE — SDK 52 compatible
// ============================================================================
import * as Notifications from 'expo-notifications';
import { AlertLevel } from '../engine/types';

Notifications.setNotificationHandler({
    handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: false,
    }),
});

export async function requestPermissions(): Promise<boolean> {
    const { status: existing } = await Notifications.getPermissionsAsync();
    if (existing === 'granted') return true;
    const { status } = await Notifications.requestPermissionsAsync();
    return status === 'granted';
}

export async function scheduleDailyReminder(timeStr: string): Promise<void> {
    await Notifications.cancelAllScheduledNotificationsAsync();
    const [hourStr, minuteStr] = timeStr.split(':');
    const hour = parseInt(hourStr, 10);
    const minute = parseInt(minuteStr, 10);

    await Notifications.scheduleNotificationAsync({
        content: {
            title: '🧠 Mental Health Check',
            body: "Time to log today's data and check your wellbeing score.",
            sound: true,
        },
        trigger: {
            hour,
            minute,
            repeats: true,
        } as any,
    });
}

export async function sendAlertNotification(
    alertLevel: AlertLevel,
    anomalyScore: number,
    flaggedFeatures: string[]
): Promise<void> {
    if (alertLevel !== 'orange' && alertLevel !== 'red') return;
    const emoji = alertLevel === 'red' ? '🚨' : '⚠️';
    const topFlags = flaggedFeatures.slice(0, 3).join(', ') || 'multiple features';
    await Notifications.scheduleNotificationAsync({
        content: {
            title: `${emoji} Alert: ${alertLevel.toUpperCase()} Level`,
            body: `Anomaly score: ${(anomalyScore * 100).toFixed(0)}%. Flagged: ${topFlags}. Consider speaking with someone.`,
            sound: true,
        },
        trigger: null,
    });
}
