// src/screens/CheckInScreen.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  SafeAreaView,
} from 'react-native';
import { useAppData } from '../../App';

type FieldConfig = {
  key: string;
  label: string;
  description: string;
  emoji: string;
  defaultValue: string;
  unit: string;
  min: number;
  max: number;
};

const FIELDS: FieldConfig[] = [
  {
    key: 'screen_time_hours',
    label: 'Screen Time',
    description: 'Total hours on your phone/computer today',
    emoji: '📱',
    defaultValue: '4.5',
    unit: 'hrs',
    min: 0,
    max: 24,
  },
  {
    key: 'sleep_duration_hours',
    label: 'Sleep Duration',
    description: 'Hours slept last night',
    emoji: '😴',
    defaultValue: '7.5',
    unit: 'hrs',
    min: 0,
    max: 24,
  },
  {
    key: 'daily_displacement_km',
    label: 'Movement',
    description: 'Approximate distance walked or traveled (km)',
    emoji: '🚶',
    defaultValue: '10',
    unit: 'km',
    min: 0,
    max: 500,
  },
  {
    key: 'texts_per_day',
    label: 'Messages Sent',
    description: 'Number of texts / chats sent today',
    emoji: '💬',
    defaultValue: '20',
    unit: 'msgs',
    min: 0,
    max: 9999,
  },
  {
    key: 'voice_energy_mean',
    label: 'Energy & Mood',
    description: 'How engaged and energetic do you feel? (1 = very low, 5 = very high)',
    emoji: '⚡',
    defaultValue: '3',
    unit: '/5',
    min: 1,
    max: 5,
  },
];

export default function CheckInScreen() {
  const { addCheckIn, checkIns } = useAppData();
  const today = new Date().toISOString().slice(0, 10);
  const alreadyCheckedIn = checkIns.some((c) => c.date === today);

  const initialValues: Record<string, string> = {};
  FIELDS.forEach((f) => {
    initialValues[f.key] = f.defaultValue;
  });

  const [values, setValues] = useState<Record<string, string>>(initialValues);

  const toNumber = (val: string, fallback: number): number => {
    const n = Number(val.replace(',', '.'));
    return Number.isFinite(n) ? n : fallback;
  };

  const onSubmit = async () => {
    // Validate ranges
    for (const field of FIELDS) {
      const n = toNumber(values[field.key], parseFloat(field.defaultValue));
      if (n < field.min || n > field.max) {
        Alert.alert(
          'Invalid Input',
          `${field.label} must be between ${field.min} and ${field.max}${field.unit}.`,
        );
        return;
      }
    }

    await addCheckIn({
      screen_time_hours: toNumber(values['screen_time_hours'], 4.5),
      sleep_duration_hours: toNumber(values['sleep_duration_hours'], 7.5),
      daily_displacement_km: toNumber(values['daily_displacement_km'], 10),
      texts_per_day: toNumber(values['texts_per_day'], 20),
      voice_energy_mean: toNumber(values['voice_energy_mean'], 3),
    });

    Alert.alert(
      '✅ Saved',
      'Your daily check-in has been recorded and analyzed.',
      [{ text: 'Great!', style: 'default' }],
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.header}>Daily Check-in</Text>
        <Text style={styles.subheader}>{today}</Text>
        {alreadyCheckedIn && (
          <View style={styles.alreadyBanner}>
            <Text style={styles.alreadyText}>
              ✏️ You already checked in today. Submitting will update today's entry.
            </Text>
          </View>
        )}
        <Text style={styles.intro}>
          Fill this once a day. MoodGuard compares your responses to your personal
          baseline to detect early signs of mood changes.
        </Text>

        {FIELDS.map((field) => (
          <View key={field.key} style={styles.fieldCard}>
            <View style={styles.fieldHeader}>
              <Text style={styles.fieldEmoji}>{field.emoji}</Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.fieldLabel}>{field.label}</Text>
                <Text style={styles.fieldDesc}>{field.description}</Text>
              </View>
              <View style={styles.unitTag}>
                <Text style={styles.unitText}>{field.unit}</Text>
              </View>
            </View>
            <TextInput
              value={values[field.key]}
              onChangeText={(v) =>
                setValues((prev) => ({ ...prev, [field.key]: v }))
              }
              keyboardType="numeric"
              style={styles.input}
              placeholder={field.defaultValue}
              placeholderTextColor="#9CA3AF"
            />
          </View>
        ))}

        <TouchableOpacity style={styles.button} onPress={onSubmit} activeOpacity={0.85}>
          <Text style={styles.buttonText}>💾  Save Today's Check-in</Text>
        </TouchableOpacity>

        <Text style={styles.footerNote}>
          All data is stored only on your device. Nothing is sent to any server.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F7F8FC',
  },
  container: {
    padding: 20,
    paddingBottom: 50,
  },
  header: {
    fontSize: 30,
    fontWeight: '800',
    color: '#1A1A2E',
    letterSpacing: -0.5,
  },
  subheader: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 12,
    marginTop: 2,
  },
  alreadyBanner: {
    backgroundColor: '#EEF2FF',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
  },
  alreadyText: {
    color: '#4F46E5',
    fontSize: 13,
    fontWeight: '500',
  },
  intro: {
    fontSize: 14,
    color: '#6B7280',
    lineHeight: 20,
    marginBottom: 20,
  },
  fieldCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  fieldHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  fieldEmoji: {
    fontSize: 22,
    marginTop: 2,
  },
  fieldLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1A1A2E',
  },
  fieldDesc: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
    lineHeight: 16,
  },
  unitTag: {
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    alignSelf: 'flex-start',
    marginTop: 2,
  },
  unitText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#6B7280',
  },
  input: {
    backgroundColor: '#F9FAFB',
    borderWidth: 1.5,
    borderColor: '#E5E7EB',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 18,
    fontWeight: '600',
    color: '#1A1A2E',
  },
  button: {
    backgroundColor: '#6366F1',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
    shadowColor: '#6366F1',
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 4,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  footerNote: {
    fontSize: 12,
    color: '#9CA3AF',
    textAlign: 'center',
    marginTop: 16,
    lineHeight: 18,
  },
});
