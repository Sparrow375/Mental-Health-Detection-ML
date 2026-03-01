import React, { useCallback, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { ALERT_BG, ALERT_COLORS, ALERT_LABELS, Colors } from '../theme';
import { DailyReport, FinalPrediction } from '../engine/types';
import { getDailyReports, getBaseline, getDetectorState, getPatientId } from '../services/storage';
import { generateFinalPrediction } from '../engine/anomalyDetector';

function ScoreGauge({ score }: { score: number }) {
    const pct = Math.round(score * 100);
    const color = pct < 35 ? Colors.green : pct < 50 ? Colors.yellow : pct < 65 ? Colors.orange : Colors.red;
    return (
        <View style={styles.gaugeWrapper}>
            <View style={[styles.gaugeOuter, { borderColor: color }]}>
                <View style={styles.gaugeInner}>
                    <Text style={[styles.gaugeScore, { color }]}>{pct}</Text>
                    <Text style={styles.gaugeLabel}>/ 100</Text>
                </View>
            </View>
            <Text style={styles.gaugeTitle}>Anomaly Score</Text>
        </View>
    );
}

export default function HomeScreen() {
    const navigation = useNavigation<any>();
    const [reports, setReports] = useState<DailyReport[]>([]);
    const [prediction, setPrediction] = useState<FinalPrediction | null>(null);
    const [loading, setLoading] = useState(true);

    const load = useCallback(async () => {
        setLoading(true);
        const [rpts, baseline, pid] = await Promise.all([getDailyReports(), getBaseline(), getPatientId()]);
        setReports(rpts);
        if (rpts.length > 0) {
            const state = await getDetectorState(baseline);
            setPrediction(generateFinalPrediction(pid, rpts.length, state));
        }
        setLoading(false);
    }, []);

    useFocusEffect(useCallback(() => { load(); }, [load]));

    const latest = reports[reports.length - 1];
    const alertLevel = latest?.alertLevel ?? 'green';
    const score = latest?.anomalyScore ?? 0;

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}
            refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={Colors.primary} />}>
            <Text style={styles.heading}>Mental Health Monitor</Text>
            <Text style={styles.sub}>{reports.length === 0 ? 'No data yet — log your first entry!' : `${reports.length} day${reports.length > 1 ? 's' : ''} monitored`}</Text>

            <View style={styles.card}>
                <ScoreGauge score={score} />
                <View style={[styles.alertBadge, { backgroundColor: ALERT_BG[alertLevel] }]}>
                    <Text style={[styles.alertText, { color: ALERT_COLORS[alertLevel] }]}>{ALERT_LABELS[alertLevel]}</Text>
                </View>
            </View>

            {latest && (
                <View style={styles.statsRow}>
                    <View style={styles.statCard}>
                        <Text style={styles.statValue}>{latest.sustainedDeviationDays}</Text>
                        <Text style={styles.statLabel}>Sustained Days</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Text style={styles.statValue}>{latest.evidenceAccumulated.toFixed(2)}</Text>
                        <Text style={styles.statLabel}>Evidence</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Text style={styles.statValue} numberOfLines={1}>{latest.patternType.replace('_', ' ')}</Text>
                        <Text style={styles.statLabel}>Pattern</Text>
                    </View>
                </View>
            )}

            {latest && latest.notes !== 'Normal operation' && (
                <View style={[styles.card, { alignItems: 'flex-start' }]}>
                    <Text style={styles.notesTitle}>Today's Notes</Text>
                    <Text style={styles.notesText}>{latest.notes}</Text>
                </View>
            )}

            {prediction && (
                <View style={[styles.card, { alignItems: 'flex-start' }]}>
                    <Text style={styles.predTitle}>{prediction.sustainedAnomalyDetected ? '⚠️ Anomaly Detected' : '✅ Monitoring Normal'}</Text>
                    <Text style={styles.predRec}>{prediction.recommendation}</Text>
                    <Text style={styles.predConf}>Confidence: {(prediction.confidence * 100).toFixed(0)}%</Text>
                </View>
            )}

            <TouchableOpacity style={styles.logBtn} onPress={() => navigation.navigate('LogData')}>
                <Text style={styles.logBtnText}>+ Log Today's Data</Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    content: { padding: 20, paddingBottom: 40 },
    heading: { fontSize: 26, fontWeight: '700', color: Colors.text, marginBottom: 4 },
    sub: { fontSize: 14, color: Colors.textMuted, marginBottom: 20 },
    card: { backgroundColor: Colors.card, borderRadius: 16, padding: 20, marginBottom: 16, alignItems: 'center' },
    gaugeWrapper: { alignItems: 'center', marginBottom: 12 },
    gaugeOuter: { width: 130, height: 130, borderRadius: 65, borderWidth: 8, alignItems: 'center', justifyContent: 'center' },
    gaugeInner: { alignItems: 'center' },
    gaugeScore: { fontSize: 38, fontWeight: '800' },
    gaugeLabel: { fontSize: 13, color: Colors.textMuted },
    gaugeTitle: { fontSize: 14, color: Colors.textMuted, marginTop: 8 },
    alertBadge: { paddingHorizontal: 20, paddingVertical: 8, borderRadius: 20, marginTop: 4 },
    alertText: { fontSize: 16, fontWeight: '700' },
    statsRow: { flexDirection: 'row', gap: 10, marginBottom: 16 },
    statCard: { flex: 1, backgroundColor: Colors.card, borderRadius: 14, padding: 14, alignItems: 'center' },
    statValue: { fontSize: 18, fontWeight: '700', color: Colors.text },
    statLabel: { fontSize: 11, color: Colors.textMuted, marginTop: 4, textAlign: 'center' },
    notesTitle: { fontSize: 13, color: Colors.textMuted, marginBottom: 6 },
    notesText: { fontSize: 14, color: Colors.text, lineHeight: 20 },
    predTitle: { fontSize: 16, fontWeight: '700', color: Colors.text, marginBottom: 8 },
    predRec: { fontSize: 13, color: Colors.textMuted, lineHeight: 20, marginBottom: 8 },
    predConf: { fontSize: 12, color: Colors.primaryLight },
    logBtn: { backgroundColor: Colors.primary, borderRadius: 14, paddingVertical: 16, alignItems: 'center', marginTop: 4 },
    logBtnText: { fontSize: 16, fontWeight: '700', color: '#fff' },
});
