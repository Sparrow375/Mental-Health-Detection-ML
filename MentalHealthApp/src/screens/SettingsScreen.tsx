import React, { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Switch, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { Colors } from '../theme';
import { getBaseline, getNotifyTime, getPatientId, markSetupDone, resetAllData, saveBaseline, saveNotifyTime, savePatientId } from '../services/storage';
import { buildBaselineFromInputs, generateSyntheticBaseline } from '../engine/baselineGenerator';
import { requestPermissions, scheduleDailyReminder } from '../services/notifications';
import { FEATURE_DEFAULTS, FEATURE_KEYS, FEATURE_LABELS } from '../engine/types';

export default function SettingsScreen() {
    const [patientId, setPatientId] = useState('PT-001');
    const [notifyTime, setNotifyTime] = useState('20:00');
    const [notifyEnabled, setNotifyEnabled] = useState(false);
    const [showBaselineEditor, setShowBaselineEditor] = useState(false);
    const [baselineInputs, setBaselineInputs] = useState<Record<string, string>>({});

    useEffect(() => {
        (async () => {
            const [pid, nt] = await Promise.all([getPatientId(), getNotifyTime()]);
            setPatientId(pid); setNotifyTime(nt);
            const init: Record<string, string> = {};
            for (const k of FEATURE_KEYS) init[k] = String((FEATURE_DEFAULTS as any)[k]);
            setBaselineInputs(init);
        })();
    }, []);

    const handleToggleNotify = async (enabled: boolean) => {
        setNotifyEnabled(enabled);
        if (enabled) {
            const granted = await requestPermissions();
            if (!granted) { Alert.alert('Permission denied', 'Enable notifications in device settings.'); setNotifyEnabled(false); return; }
            await scheduleDailyReminder(notifyTime);
            Alert.alert('Scheduled', `Daily reminder set for ${notifyTime}`);
        }
    };

    const handleSaveNotifyTime = async () => {
        if (!/^\d{2}:\d{2}$/.test(notifyTime)) { Alert.alert('Invalid', 'Use HH:MM format e.g. 20:00'); return; }
        await saveNotifyTime(notifyTime);
        if (notifyEnabled) await scheduleDailyReminder(notifyTime);
        Alert.alert('Saved', `Reminder set to ${notifyTime}`);
    };

    const handleSaveCustomBaseline = async () => {
        const parsed: Record<string, number> = {};
        for (const k of FEATURE_KEYS) {
            const val = parseFloat(baselineInputs[k] ?? '');
            if (isNaN(val)) { Alert.alert('Validation', `"${FEATURE_LABELS[k]}" is not valid.`); return; }
            parsed[k] = val;
        }
        await saveBaseline(buildBaselineFromInputs(parsed));
        await markSetupDone();
        setShowBaselineEditor(false);
        Alert.alert('Saved', 'Custom baseline saved.');
    };

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <Text style={styles.heading}>Settings</Text>

            <View style={styles.card}>
                <Text style={styles.cardTitle}>Patient ID</Text>
                <TextInput style={styles.input} value={patientId} onChangeText={setPatientId} placeholderTextColor={Colors.textMuted} />
                <TouchableOpacity style={styles.btn} onPress={async () => { await savePatientId(patientId.trim() || 'PT-001'); Alert.alert('Saved', 'Patient ID updated.'); }}>
                    <Text style={styles.btnText}>Save ID</Text>
                </TouchableOpacity>
            </View>

            <View style={styles.card}>
                <Text style={styles.cardTitle}>Notifications</Text>
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <Text style={{ fontSize: 14, color: Colors.text }}>Daily Reminder</Text>
                    <Switch value={notifyEnabled} onValueChange={handleToggleNotify} trackColor={{ true: Colors.primary }} thumbColor="#fff" />
                </View>
                {notifyEnabled && (
                    <>
                        <Text style={{ fontSize: 12, color: Colors.textMuted, marginBottom: 8 }}>Reminder time (HH:MM)</Text>
                        <TextInput style={styles.input} value={notifyTime} onChangeText={setNotifyTime} placeholderTextColor={Colors.textMuted} />
                        <TouchableOpacity style={styles.btn} onPress={handleSaveNotifyTime}><Text style={styles.btnText}>Save Time</Text></TouchableOpacity>
                    </>
                )}
            </View>

            <View style={styles.card}>
                <Text style={styles.cardTitle}>Baseline Setup</Text>
                <Text style={{ fontSize: 12, color: Colors.textMuted, marginBottom: 12 }}>Your baseline defines your "normal" behavior. The detector measures deviations from it.</Text>
                <TouchableOpacity style={[styles.btn, { backgroundColor: Colors.surface, marginBottom: 10 }]}
                    onPress={() => Alert.alert('Use Synthetic Baseline', 'Replace baseline with typical defaults?', [
                        { text: 'Cancel', style: 'cancel' },
                        { text: 'Generate', onPress: async () => { await saveBaseline(generateSyntheticBaseline()); await markSetupDone(); Alert.alert('Done', 'Synthetic baseline saved.'); } }
                    ])}>
                    <Text style={[styles.btnText, { color: Colors.primaryLight }]}>✨ Use Synthetic Default Baseline</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.btn, { backgroundColor: Colors.surface }]} onPress={() => setShowBaselineEditor(v => !v)}>
                    <Text style={[styles.btnText, { color: Colors.primaryLight }]}>{showBaselineEditor ? '▲ Hide Editor' : '✏️ Enter Custom Baseline'}</Text>
                </TouchableOpacity>
                {showBaselineEditor && (
                    <View style={{ marginTop: 12 }}>
                        {FEATURE_KEYS.map(k => (
                            <View key={k} style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                                <Text style={{ flex: 1, fontSize: 12, color: Colors.textMuted }}>{FEATURE_LABELS[k]}</Text>
                                <TextInput style={[styles.input, { width: 80, marginBottom: 0, padding: 8, fontSize: 13 }]}
                                    value={baselineInputs[k] ?? ''} onChangeText={t => setBaselineInputs(p => ({ ...p, [k]: t }))}
                                    keyboardType="decimal-pad" placeholderTextColor={Colors.textMuted} />
                            </View>
                        ))}
                        <TouchableOpacity style={[styles.btn, { marginTop: 8 }]} onPress={handleSaveCustomBaseline}><Text style={styles.btnText}>Save Custom Baseline</Text></TouchableOpacity>
                    </View>
                )}
            </View>

            <View style={[styles.card, { borderWidth: 1, borderColor: Colors.red + '44' }]}>
                <Text style={[styles.cardTitle, { color: Colors.red }]}>Danger Zone</Text>
                <TouchableOpacity style={[styles.btn, { backgroundColor: Colors.redDim }]}
                    onPress={() => Alert.alert('Reset All Data', 'This will permanently delete all logs. Are you sure?', [
                        { text: 'Cancel', style: 'cancel' },
                        { text: 'Reset', style: 'destructive', onPress: async () => { await resetAllData(); Alert.alert('Done', 'All data has been cleared.'); } }
                    ])}>
                    <Text style={[styles.btnText, { color: Colors.red }]}>🗑 Reset All Data</Text>
                </TouchableOpacity>
            </View>

            <Text style={{ textAlign: 'center', fontSize: 12, color: Colors.textMuted, marginTop: 8 }}>Mental Health Detection v1.0 · Local-only · No data leaves your device.</Text>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.bg },
    content: { padding: 20, paddingBottom: 60 },
    heading: { fontSize: 24, fontWeight: '700', color: Colors.text, marginBottom: 16 },
    card: { backgroundColor: Colors.card, borderRadius: 16, padding: 16, marginBottom: 16 },
    cardTitle: { fontSize: 15, fontWeight: '700', color: Colors.text, marginBottom: 12 },
    input: { backgroundColor: Colors.surface, borderRadius: 10, padding: 12, color: Colors.text, fontSize: 15, borderWidth: 1, borderColor: Colors.border, marginBottom: 10, textAlign: 'right' },
    btn: { backgroundColor: Colors.primary, borderRadius: 10, padding: 12, alignItems: 'center' },
    btnText: { fontSize: 14, fontWeight: '700', color: '#fff' },
});
