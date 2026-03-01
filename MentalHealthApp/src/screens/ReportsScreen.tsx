import React, { useCallback, useState } from 'react';
import { Dimensions, FlatList, RefreshControl, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { ALERT_COLORS, ALERT_LABELS, Colors } from '../theme';
import { DailyReport } from '../engine/types';
import { getDailyReports } from '../services/storage';

const { width: SW } = Dimensions.get('window');
const CHART_W = SW - 40;
const CHART_H = 140;
type Tab = 'daily' | 'weekly' | 'history';

function MiniChart({ values, colors }: { values: number[]; colors: string[] }) {
    if (values.length === 0) return null;
    const barW = Math.max(2, (CHART_W - 20) / values.length - 1);
    return (
        <View style={{ width: CHART_W, height: CHART_H, marginVertical: 8 }}>
            {[0, 0.35, 0.65].map(v => (
                <View key={v} style={{ position: 'absolute', top: CHART_H * (1 - v), left: 0, right: 0, height: 1, backgroundColor: v === 0.65 ? Colors.orange + '44' : Colors.border }} />
            ))}
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', height: CHART_H }}>
                {values.map((val, i) => (
                    <View key={i} style={{ width: barW, height: Math.max(2, val * (CHART_H - 4)), backgroundColor: colors[i] ?? Colors.primary, borderRadius: 2, marginRight: 1 }} />
                ))}
            </View>
        </View>
    );
}

function DailyRow({ report }: { report: DailyReport }) {
    const date = new Date(report.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const color = ALERT_COLORS[report.alertLevel];
    return (
        <View style={[styles.row, { borderLeftColor: color }]}>
            <View style={{ flex: 1 }}>
                <Text style={styles.rowDate}>Day {report.dayNumber}  ·  {date}</Text>
                <Text style={styles.rowPattern}>{report.patternType.replace('_', ' ')}</Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
                <Text style={[styles.rowScore, { color }]}>{(report.anomalyScore * 100).toFixed(0)}</Text>
                <Text style={styles.rowScoreLabel}>score</Text>
                <Text style={[{ fontSize: 12, fontWeight: '600', marginTop: 2 }, { color }]}>{ALERT_LABELS[report.alertLevel]}</Text>
            </View>
        </View>
    );
}

function weeklyBuckets(reports: DailyReport[]) {
    return Array.from({ length: Math.ceil(reports.length / 7) }, (_, i) => {
        const chunk = reports.slice(i * 7, i * 7 + 7);
        const avg = chunk.reduce((s, r) => s + r.anomalyScore, 0) / chunk.length;
        const order: Record<string, number> = { green: 0, yellow: 1, orange: 2, red: 3 };
        const maxAlert = chunk.reduce((m, r) => order[r.alertLevel] > order[m] ? r.alertLevel : m, 'green');
        return { label: `Wk ${i + 1}`, avg, color: ALERT_COLORS[maxAlert] };
    });
}

export default function ReportsScreen() {
    const [tab, setTab] = useState<Tab>('daily');
    const [reports, setReports] = useState<DailyReport[]>([]);
    const [loading, setLoading] = useState(true);

    useFocusEffect(useCallback(() => {
        (async () => { setLoading(true); setReports(await getDailyReports()); setLoading(false); })();
    }, []));

    const weeks = weeklyBuckets(reports);

    return (
        <View style={styles.container}>
            <View style={styles.tabs}>
                {(['daily', 'weekly', 'history'] as Tab[]).map(t => (
                    <TouchableOpacity key={t} style={[styles.tab, tab === t && styles.tabActive]} onPress={() => setTab(t)}>
                        <Text style={[styles.tabText, tab === t && styles.tabTextActive]}>{t === 'daily' ? 'Daily' : t === 'weekly' ? 'Weekly' : 'History'}</Text>
                    </TouchableOpacity>
                ))}
            </View>

            {reports.length === 0 && !loading && (
                <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
                    <Text style={{ color: Colors.textMuted, fontSize: 15 }}>No data yet. Log your first entry.</Text>
                </View>
            )}

            {tab === 'daily' && (
                <FlatList data={[...reports].reverse()} keyExtractor={r => String(r.dayNumber)}
                    renderItem={({ item }) => <DailyRow report={item} />} contentContainerStyle={{ padding: 16 }}
                    refreshControl={<RefreshControl refreshing={loading} tintColor={Colors.primary} onRefresh={() => { }} />} />
            )}

            {tab === 'weekly' && (
                <ScrollView contentContainerStyle={{ padding: 16 }}>
                    <Text style={styles.sectionTitle}>Weekly Average Anomaly Score</Text>
                    <MiniChart values={weeks.map(w => w.avg)} colors={weeks.map(w => w.color)} />
                    <View style={{ flexDirection: 'row', justifyContent: 'space-around', marginBottom: 12 }}>
                        {weeks.map(w => <Text key={w.label} style={{ fontSize: 11, color: Colors.textMuted }}>{w.label}</Text>)}
                    </View>
                    {weeks.map((w, i) => (
                        <View key={i} style={[styles.row, { borderLeftColor: w.color }]}>
                            <Text style={styles.rowDate}>{w.label}</Text>
                            <Text style={[styles.rowScore, { color: w.color, fontSize: 14 }]}>{(w.avg * 100).toFixed(0)} avg score</Text>
                        </View>
                    ))}
                </ScrollView>
            )}

            {tab === 'history' && (
                <ScrollView contentContainerStyle={{ padding: 16 }}>
                    <Text style={styles.sectionTitle}>Full Score History</Text>
                    {reports.length > 0 && <MiniChart values={reports.map(r => r.anomalyScore)} colors={reports.map(r => ALERT_COLORS[r.alertLevel])} />}
                    <Text style={{ color: Colors.textMuted, fontSize: 13, marginBottom: 16 }}>
                        {reports.length} entries · Max: {reports.length > 0 ? (Math.max(...reports.map(r => r.anomalyScore)) * 100).toFixed(0) + '%' : 'N/A'}
                    </Text>
                    <Text style={styles.sectionTitle}>Alert Distribution</Text>
                    {(['green', 'yellow', 'orange', 'red'] as const).map(lvl => {
                        const count = reports.filter(r => r.alertLevel === lvl).length;
                        const pct = reports.length > 0 ? (count / reports.length) * 100 : 0;
                        return (
                            <View key={lvl} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 8 }}>
                                <Text style={{ width: 90, fontSize: 12, fontWeight: '600', color: ALERT_COLORS[lvl] }}>{ALERT_LABELS[lvl]}</Text>
                                <View style={{ flex: 1, height: 12, backgroundColor: Colors.surface, borderRadius: 6, overflow: 'hidden' }}>
                                    <View style={{ width: `${pct}%`, height: 12, backgroundColor: ALERT_COLORS[lvl], borderRadius: 6 }} />
                                </View>
                                <Text style={{ width: 32, fontSize: 12, color: Colors.textMuted, textAlign: 'right' }}>{count}d</Text>
                            </View>
                        );
                    })}
                </ScrollView>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    tabs: { flexDirection: 'row', backgroundColor: Colors.surface, padding: 6, margin: 16, marginBottom: 0, borderRadius: 12 },
    tab: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 8 },
    tabActive: { backgroundColor: Colors.primary },
    tabText: { fontSize: 14, color: Colors.textMuted, fontWeight: '600' },
    tabTextActive: { color: '#fff' },
    row: { backgroundColor: Colors.card, borderRadius: 12, padding: 14, marginBottom: 10, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderLeftWidth: 4 },
    rowDate: { fontSize: 14, fontWeight: '600', color: Colors.text },
    rowPattern: { fontSize: 12, color: Colors.textMuted, marginTop: 4, textTransform: 'capitalize' },
    rowScore: { fontSize: 24, fontWeight: '800' },
    rowScoreLabel: { fontSize: 11, color: Colors.textMuted },
    sectionTitle: { fontSize: 15, fontWeight: '700', color: Colors.text, marginBottom: 8, marginTop: 8 },
});
