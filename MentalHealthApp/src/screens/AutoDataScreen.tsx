import React, { useCallback, useState } from 'react';
import {
    ActivityIndicator, Alert, ScrollView, StyleSheet, Text,
    TouchableOpacity, View, RefreshControl, Switch,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Colors } from '../theme';
import { FEATURE_LABELS, FEATURE_KEYS, FEATURE_DEFAULTS, FEATURE_RANGES } from '../engine/types';
import { collectTodayData, getTodaySnapshot, saveTodaySnapshot, storeLocationSnapshot, recordBatterySnapshot } from '../services/autoCollector';
import { analyze } from '../engine/anomalyDetector';
import { appendDailyReport, appendDeviationsHistory, getBaseline, getDailyReports, getDetectorState, saveDetectorState } from '../services/storage';
import { sendAlertNotification } from '../services/notifications';

// Features that are auto-collected (vs baseline defaults)
const AUTO_FEATURES = new Set([
    'screen_time_hours', 'unlock_count', 'wake_time_hour', 'sleep_time_hour',
    'dark_duration_hours', 'daily_displacement_km', 'location_entropy',
    'home_time_ratio', 'places_visited', 'charge_duration_hours', 'unique_contacts',
    'sleep_duration_hours', 'social_app_ratio',
]);

const SECTIONS = [
    { title: '📱 Digital Activity', keys: ['screen_time_hours', 'unlock_count', 'social_app_ratio'] },
    { title: '💬 Social Connection', keys: ['calls_per_day', 'texts_per_day', 'unique_contacts', 'response_time_minutes'] },
    { title: '🚶 Movement & Mobility', keys: ['daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited'] },
    { title: '🌙 Circadian & Sleep', keys: ['wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours', 'dark_duration_hours', 'charge_duration_hours'] },
    { title: '🗣 Conversation', keys: ['conversation_duration_hours', 'conversation_frequency'] },
];

function formatValue(key: string, val: number): string {
    if (['social_app_ratio', 'home_time_ratio'].includes(key)) return val.toFixed(2);
    if (['daily_displacement_km', 'location_entropy', 'screen_time_hours',
        'sleep_duration_hours', 'dark_duration_hours', 'charge_duration_hours',
        'conversation_duration_hours'].includes(key)) return val.toFixed(1);
    return val.toFixed(0);
}

export default function AutoDataScreen() {
    const navigation = useNavigation<any>();
    const [data, setData] = useState<Record<string, number> | null>(null);
    const [lastTs, setLastTs] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);
    const [analyzing, setAnalyzing] = useState(false);
    const [showOverride, setShowOverride] = useState(false);

    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            // always show freshest data
            await storeLocationSnapshot();
            await recordBatterySnapshot();
            const collected = await collectTodayData();
            await saveTodaySnapshot(collected);
            setData(collected);
            setLastTs(Date.now());
        } catch {
            // fallback: show last saved snapshot
            const snap = await getTodaySnapshot();
            if (snap) { setData(snap.data); setLastTs(snap.ts); }
        } finally {
            setLoading(false);
        }
    }, []);

    useFocusEffect(useCallback(() => { refresh(); }, [refresh]));

    const handleAnalyze = async () => {
        if (!data) return;
        setAnalyzing(true);
        try {
            const [baseline, existing] = await Promise.all([getBaseline(), getDailyReports()]);
            const state = await getDetectorState(baseline);
            const deviationsHistory = existing.map(r => r.topDeviations);
            const { anomalyReport, dailyReport, newState } = analyze(
                data, baseline, state, deviationsHistory, existing.length + 1, data as any
            );
            await Promise.all([
                appendDailyReport(dailyReport),
                saveDetectorState(newState),
                appendDeviationsHistory(anomalyReport.featureDeviations),
            ]);
            await sendAlertNotification(dailyReport.alertLevel, dailyReport.anomalyScore, dailyReport.flaggedFeatures);
            Alert.alert(
                'Analysis Complete',
                `Alert Level: ${dailyReport.alertLevel.toUpperCase()}\nAnomaly Score: ${(dailyReport.anomalyScore * 100).toFixed(0)}/100`,
                [{ text: 'View Report', onPress: () => navigation.navigate('Home') }]
            );
        } catch {
            Alert.alert('Error', 'Analysis failed. Please try again.');
        } finally {
            setAnalyzing(false);
        }
    };

    const updateValue = (key: string, delta: number) => {
        if (!data) return;
        const range = FEATURE_RANGES[key];
        const step = ['screen_time_hours', 'sleep_duration_hours', 'dark_duration_hours',
            'charge_duration_hours', 'conversation_duration_hours', 'daily_displacement_km',
            'location_entropy'].includes(key) ? 0.5 :
            ['social_app_ratio', 'home_time_ratio'].includes(key) ? 0.05 : 1;
        const newVal = Math.max(range.min, Math.min(range.max, (data[key] ?? 0) + delta * step));
        setData(prev => prev ? { ...prev, [key]: parseFloat(newVal.toFixed(2)) } : prev);
    };

    if (loading && !data) {
        return <View style={styles.centered}><ActivityIndicator size="large" color={Colors.primary} /><Text style={styles.loadingText}>Collecting sensor data…</Text></View>;
    }

    const tsLabel = lastTs
        ? `Last updated: ${new Date(lastTs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
        : 'Collecting…';

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}
            refreshControl={<RefreshControl refreshing={loading} onRefresh={refresh} tintColor={Colors.primary} />}>

            {/* Status Header */}
            <View style={styles.statusBanner}>
                <View style={styles.statusDot} />
                <Text style={styles.statusText}>🟢 Auto-monitoring active</Text>
                <Text style={styles.tsText}>{tsLabel}</Text>
            </View>

            {/* Data Preview Cards */}
            {SECTIONS.map(section => (
                <View key={section.title} style={styles.section}>
                    <Text style={styles.sectionTitle}>{section.title}</Text>
                    {section.keys.map(k => {
                        const val = data?.[k] ?? (FEATURE_DEFAULTS as any)[k];
                        const isAuto = AUTO_FEATURES.has(k);
                        return (
                            <View key={k} style={styles.row}>
                                <View style={styles.rowLeft}>
                                    <Text style={styles.rowLabel}>{FEATURE_LABELS[k]}</Text>
                                    <View style={[styles.badge, { backgroundColor: isAuto ? Colors.primary + '22' : Colors.surface }]}>
                                        <Text style={[styles.badgeText, { color: isAuto ? Colors.primary : Colors.textMuted }]}>
                                            {isAuto ? '📡 auto' : '⚙️ default'}
                                        </Text>
                                    </View>
                                </View>
                                <View style={styles.rowRight}>
                                    {showOverride && (
                                        <TouchableOpacity style={styles.adjBtn} onPress={() => updateValue(k, -1)}>
                                            <Text style={styles.adjText}>−</Text>
                                        </TouchableOpacity>
                                    )}
                                    <Text style={styles.rowValue}>{formatValue(k, val)}</Text>
                                    {showOverride && (
                                        <TouchableOpacity style={styles.adjBtn} onPress={() => updateValue(k, 1)}>
                                            <Text style={styles.adjText}>+</Text>
                                        </TouchableOpacity>
                                    )}
                                </View>
                            </View>
                        );
                    })}
                </View>
            ))}

            {/* Override Toggle */}
            <View style={styles.overrideRow}>
                <Text style={{ color: Colors.textMuted, fontSize: 13 }}>Manual override mode</Text>
                <Switch value={showOverride} onValueChange={setShowOverride}
                    trackColor={{ true: Colors.primary }} thumbColor="#fff" />
            </View>

            {/* Analyze Button */}
            <TouchableOpacity style={[styles.analyzeBtn, analyzing && { opacity: 0.6 }]}
                onPress={handleAnalyze} disabled={analyzing || !data}>
                <Text style={styles.analyzeBtnText}>
                    {analyzing ? '⏳ Analyzing…' : '▶ Run Analysis Now'}
                </Text>
            </TouchableOpacity>

            <Text style={styles.hint}>
                Data is collected automatically throughout the day.{'\n'}
                Pull to refresh. Analysis runs automatically at midnight.
            </Text>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    content: { padding: 16, paddingBottom: 100 },
    centered: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: Colors.bg, gap: 12 },
    loadingText: { color: Colors.textMuted, fontSize: 14 },
    statusBanner: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.card, borderRadius: 14, padding: 14, marginBottom: 16, gap: 8, flexWrap: 'wrap' },
    statusDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#4CAF50' },
    statusText: { fontSize: 14, fontWeight: '600', color: Colors.text, flex: 1 },
    tsText: { fontSize: 11, color: Colors.textMuted },
    section: { backgroundColor: Colors.card, borderRadius: 16, padding: 14, marginBottom: 14 },
    sectionTitle: { fontSize: 14, fontWeight: '700', color: Colors.primary, marginBottom: 10 },
    row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: Colors.border + '55' },
    rowLeft: { flex: 1, marginRight: 8 },
    rowLabel: { fontSize: 13, color: Colors.text, fontWeight: '500' },
    badge: { alignSelf: 'flex-start', borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2, marginTop: 3 },
    badgeText: { fontSize: 10, fontWeight: '600' },
    rowRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    rowValue: { fontSize: 15, fontWeight: '700', color: Colors.text, minWidth: 44, textAlign: 'right' },
    adjBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: Colors.surface, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: Colors.border },
    adjText: { fontSize: 16, fontWeight: '700', color: Colors.text },
    overrideRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: Colors.card, borderRadius: 14, padding: 14, marginBottom: 14 },
    analyzeBtn: { backgroundColor: Colors.primary, borderRadius: 14, paddingVertical: 16, alignItems: 'center', marginBottom: 12 },
    analyzeBtnText: { fontSize: 16, fontWeight: '700', color: '#fff' },
    hint: { textAlign: 'center', fontSize: 12, color: Colors.textMuted, lineHeight: 18 },
});
