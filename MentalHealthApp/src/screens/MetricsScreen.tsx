import React, { useCallback, useState } from 'react';
import { Dimensions, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { ALERT_COLORS, Colors } from '../theme';
import { DailyReport, FEATURE_LABELS, FinalPrediction } from '../engine/types';
import { getDailyReports, getBaseline, getDetectorState, getPatientId } from '../services/storage';
import { generateFinalPrediction } from '../engine/anomalyDetector';

const { width: SW } = Dimensions.get('window');
const CHART_W = SW - 40;

function EvidenceChart({ reports }: { reports: DailyReport[] }) {
    if (reports.length < 2) return null;
    const values = reports.map(r => r.evidenceAccumulated);
    const maxV = Math.max(...values, 2.1);
    const H = 100;
    const barW = Math.max(2, (CHART_W - 20) / values.length - 1);
    return (
        <View style={{ height: H + 20, marginVertical: 8 }}>
            {[0, 2.0].map(v => (
                <View key={v} style={{ position: 'absolute', top: H * (1 - v / maxV), left: 0, right: 0, height: 1, backgroundColor: v === 2.0 ? Colors.orange + '88' : Colors.border }} />
            ))}
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', height: H }}>
                {values.map((val, i) => (
                    <View key={i} style={{ width: barW, height: Math.max(2, (val / maxV) * (H - 4)), backgroundColor: val >= 2.0 ? Colors.orange : Colors.primary + 'aa', borderRadius: 2, marginRight: 1 }} />
                ))}
            </View>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: 4 }}>
                <Text style={{ fontSize: 11, color: Colors.textMuted }}>Day 1</Text>
                <Text style={{ fontSize: 11, color: Colors.orange }}>— threshold 2.0</Text>
                <Text style={{ fontSize: 11, color: Colors.textMuted }}>Day {reports.length}</Text>
            </View>
        </View>
    );
}

function MetricBox({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <View style={styles.metricBox}>
            <Text style={[styles.metricVal, { color }]}>{value}</Text>
            <Text style={styles.metricLabel}>{label}</Text>
        </View>
    );
}

export default function MetricsScreen() {
    const [reports, setReports] = useState<DailyReport[]>([]);
    const [prediction, setPrediction] = useState<FinalPrediction | null>(null);

    useFocusEffect(useCallback(() => {
        (async () => {
            const [rpts, baseline, pid] = await Promise.all([getDailyReports(), getBaseline(), getPatientId()]);
            setReports(rpts);
            if (rpts.length > 0) {
                const state = await getDetectorState(baseline);
                setPrediction(generateFinalPrediction(pid, rpts.length, state));
            }
        })();
    }, []));

    const latest = reports[reports.length - 1];
    const topDeviations = latest?.topDeviations ?? {};

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <Text style={styles.heading}>Advanced Metrics</Text>
            {reports.length === 0 && <Text style={{ color: Colors.textMuted, fontSize: 15, textAlign: 'center', marginTop: 60 }}>Log data to see metrics.</Text>}

            {prediction && (
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Final Prediction</Text>
                    <View style={styles.row2}>
                        <MetricBox label="Status" value={prediction.sustainedAnomalyDetected ? 'ANOMALY' : 'NORMAL'} color={prediction.sustainedAnomalyDetected ? Colors.red : Colors.green} />
                        <MetricBox label="Confidence" value={`${(prediction.confidence * 100).toFixed(0)}%`} color={Colors.primaryLight} />
                        <MetricBox label="Final Score" value={`${(prediction.finalAnomalyScore * 100).toFixed(0)}`} color={Colors.text} />
                    </View>
                    <Text style={{ fontSize: 13, color: Colors.textMuted, marginTop: 4, textTransform: 'capitalize' }}>Pattern: {prediction.patternIdentified}</Text>
                </View>
            )}

            {prediction && (
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Evidence Summary</Text>
                    <View style={styles.row2}>
                        <MetricBox label="Peak Evidence" value={prediction.evidenceSummary.peakEvidence.toFixed(2)} color={prediction.evidenceSummary.peakEvidence >= 2.7 ? Colors.red : Colors.yellow} />
                        <MetricBox label="Max Sustained" value={`${prediction.evidenceSummary.maxSustainedDays}d`} color={prediction.evidenceSummary.maxSustainedDays >= 5 ? Colors.orange : Colors.text} />
                        <MetricBox label="Days > Threshold" value={`${prediction.evidenceSummary.daysAboveThreshold}`} color={Colors.textMuted} />
                    </View>
                    <View style={[styles.row2, { marginTop: 8 }]}>
                        <MetricBox label="Max Score" value={`${(prediction.evidenceSummary.maxDailyAnomalyScore * 100).toFixed(0)}`} color={Colors.text} />
                        <MetricBox label="Avg Score" value={`${(prediction.evidenceSummary.avgRecentAnomalyScore * 100).toFixed(0)}`} color={Colors.text} />
                        <MetricBox label="Days Monitored" value={`${prediction.evidenceSummary.monitoringDays}`} color={Colors.primaryLight} />
                    </View>
                </View>
            )}

            {reports.length > 1 && (
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Evidence Accumulation Timeline</Text>
                    <EvidenceChart reports={reports} />
                </View>
            )}

            {latest && Object.keys(topDeviations).length > 0 && (
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Top Deviations (Latest Day)</Text>
                    <Text style={{ fontSize: 12, color: Colors.textMuted, marginBottom: 10, marginTop: -6 }}>Standard deviations from your baseline</Text>
                    {Object.entries(topDeviations).map(([k, v]) => {
                        const color = Math.abs(v) > 2.5 ? Colors.red : Math.abs(v) > 1.5 ? Colors.orange : Math.abs(v) > 0.5 ? Colors.yellow : Colors.green;
                        const pct = Math.min(Math.abs(v) / 4, 1);
                        return (
                            <View key={k} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 8 }}>
                                <Text style={{ width: 130, fontSize: 12, color: Colors.textMuted }} numberOfLines={1}>{FEATURE_LABELS[k] ?? k}</Text>
                                <View style={{ flex: 1, height: 10, backgroundColor: Colors.surface, borderRadius: 5, overflow: 'hidden' }}>
                                    <View style={{ width: `${pct * 100}%`, height: 10, backgroundColor: color, borderRadius: 5 }} />
                                </View>
                                <Text style={{ width: 40, fontSize: 12, fontWeight: '700', textAlign: 'right', color }}>{v.toFixed(2)}</Text>
                            </View>
                        );
                    })}
                </View>
            )}

            {prediction && (
                <View style={[styles.card, { borderLeftWidth: 4, borderLeftColor: prediction.sustainedAnomalyDetected ? Colors.orange : Colors.green }]}>
                    <Text style={styles.cardTitle}>Recommendation</Text>
                    <Text style={{ fontSize: 14, color: Colors.text, lineHeight: 22 }}>{prediction.recommendation}</Text>
                </View>
            )}
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    content: { padding: 20, paddingBottom: 40 },
    heading: { fontSize: 24, fontWeight: '700', color: Colors.text, marginBottom: 16 },
    card: { backgroundColor: Colors.card, borderRadius: 16, padding: 16, marginBottom: 16 },
    cardTitle: { fontSize: 15, fontWeight: '700', color: Colors.text, marginBottom: 12 },
    row2: { flexDirection: 'row', gap: 8 },
    metricBox: { flex: 1, backgroundColor: Colors.surface, borderRadius: 12, padding: 12, alignItems: 'center' },
    metricVal: { fontSize: 20, fontWeight: '800' },
    metricLabel: { fontSize: 10, color: Colors.textMuted, marginTop: 4, textAlign: 'center' },
});
