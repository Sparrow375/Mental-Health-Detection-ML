// src/utils/storage.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { DailyInput } from './system1';

const CHECKINS_KEY = 'moodguard_checkins_v1';

export async function loadCheckIns(): Promise<DailyInput[]> {
  const raw = await AsyncStorage.getItem(CHECKINS_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function saveCheckIns(data: DailyInput[]): Promise<void> {
  await AsyncStorage.setItem(CHECKINS_KEY, JSON.stringify(data));
}
