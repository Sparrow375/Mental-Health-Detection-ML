import React, { useEffect, useState } from 'react';
import { Alert, KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Colors } from '../theme';
import { FEATURE_DEFAULTS, FEATURE_KEYS, FEATURE_LABELS, FEATURE_RANGES } from '../engine/types';
import { analyze } from '../engine/anomalyDetector';
import { appendDailyReport, appendDeviationsHistory, getBaseline, getDailyReports, getDetectorState, saveDetectorState } from '../services/storage';
import { sendAlertNotification } from '../services/notifications';

const SECTIONS = [
    { title: '📱 Digital Activity', keys: ['screen_time_hours', 'unlock_count', 'social_app_ratio'] },
    { title: '💬 Social Connection', keys: ['calls_per_day', 'texts_per_day', 'unique_contacts', 'response_time_minutes'] },
    { title: '🚶 Movement & Mobility', keys: ['daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited'] },
    { title: '🌙 Circadian & Sleep', keys: ['wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours', 'dark_duration_hours', 'charge_duration_hours'] },
    { title: '🗣 Conversation', keys: ['conversation_duration_hours', 'conversation_frequency'] },
];

export default function LogDataScreen() {
    const navigation = useNavigation<any>();
    const [values, setValues] = useState<Record<string, string>>({});
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const init: Record<string, string> = {};
        for (const k of FEATURE_KEYS) init[k] = String((FEATURE_DEFAULTS as any)[k]);
        setValues(init);
    }, []);

    const handleSave = async () => {
        const parsed: Record<string, number> = {};
        for (const k of FEATURE_KEYS) {
            const val = parseFloat(values[k] ?? '');
            if (isNaN(val)) { Alert.alert('Validation Error', `"${FEATURE_LABELS[k]}" is not a valid number.`); return; }
            const range = FEATURE_RANGES[k];
            if (val < range.min || val > range.max) { Alert.alert('Validation Error', `"${FEATURE_LABELS[k]}" must be ${range.min}–${range.max}.`); return; }
            parsed[k] = val;
        }
        setSaving(true);
        try {
            const [baseline, existing] = await Promise.all([getBaseline(), getDailyReports()]);
            const state = await getDetectorState(baseline);
            const deviationsHistory = existing.map(r => r.topDeviations);
            const { anomalyReport, dailyReport, newState } = analyze(parsed, baseline, state, deviationsHistory, existing.length + 1, parsed as any);
            await Promise.all([appendDailyReport(dailyReport), saveDetectorState(newState), appendDeviationsHistory(anomalyReport.featureDeviations)]);
            await sendAlertNotification(dailyReport.alertLevel, dailyReport.anomalyScore, dailyReport.flaggedFeatures);
            navigation.navigate('Home');
        } catch (e) { Alert.alert('Error', 'Failed to save data.'); } finally { setSaving(false); }
    };

    return (
        <KeyboardAvoidingView style={{ flex: 1, backgroundColor: Colors.bg }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
            <ScrollView style={styles.container} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
                <Text style={styles.heading}>Log Today's Data</Text>
                <Text style={styles.sub}>Enter today's values. Tap Save to run analysis.</Text>
                {SECTIONS.map(section => (
                    <View key={section.title} style={styles.section}>
                        <Text style={styles.sectionTitle}>{section.title}</Text>
                        {section.keys.map(k => (
                            <View key={k} style={styles.fieldRow}>
                                <View style={styles.fieldLabelBox}>
                                    <Text style={styles.fieldLabel}>{FEATURE_LABELS[k]}</Text>
                                    <Text style={styles.fieldHint}>{FEATURE_RANGES[k].min}–{FEATURE_RANGES[k].max}</Text>
                                </View>
                                <TextInput
                                    style={styles.input}
                                    value={values[k] ?? ''}
                                    onChangeText={t => setValues(p => ({ ...p, [k]: t }))}
                                    keyboardType="decimal-pad"
                                    placeholderTextColor={Colors.textMuted}
                                    placeholder={String((FEATURE_DEFAULTS as any)[k])}
                                    selectTextOnFocus
                                />
                            </View>
                        ))}
                    </View>
                ))}
                <TouchableOpacity style={[styles.saveBtn, saving && { opacity: 0.5 }]} onPress={handleSave} disabled={saving}>
                    <Text style={styles.saveBtnText}>{saving ? 'Saving…' : '✓ Save & Analyze'}</Text>
                </TouchableOpacity>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    content: { padding: 20, paddingBottom: 60 },
    heading: { fontSize: 24, fontWeight: '700', color: Colors.text, marginBottom: 4 },
    sub: { fontSize: 13, color: Colors.textMuted, marginBottom: 20 },
    section: { backgroundColor: Colors.card, borderRadius: 16, padding: 16, marginBottom: 16 },
    sectionTitle: { fontSize: 15, fontWeight: '700', color: Colors.primary, marginBottom: 12 },
    fieldRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
    fieldLabelBox: { flex: 1, marginRight: 12 },
    fieldLabel: { fontSize: 13, color: Colors.text },
    fieldHint: { fontSize: 11, color: Colors.textMuted },
    input: { backgroundColor: Colors.surface, borderRadius: 10, padding: 10, color: Colors.text, fontSize: 15, width: 90, textAlign: 'right', borderWidth: 1, borderColor: Colors.border },
    saveBtn: { backgroundColor: Colors.primary, borderRadius: 14, paddingVertical: 16, alignItems: 'center', marginTop: 8 },
    saveBtnText: { fontSize: 16, fontWeight: '700', color: '#fff' },
});
