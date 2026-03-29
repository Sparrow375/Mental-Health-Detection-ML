import { collection, query, orderBy, limit, getDocs, doc, getDoc, where } from 'firebase/firestore';
import { db } from './config';

export interface Patient {
  id: string; // The uid
  email: string; // Stored in users/{uid} document if available
  [key: string]: any;
}

export interface MLResult {
  date: string;
  anomaly_score: number;
  prototype_match: string | null;
  match_message: string | null;
  [key: string]: any;
}

export interface DailyFeatures {
  date: string;
  screenTimeHours: number;
  socialAppRatio: number;
  dailyDisplacementKm: number;
  placesVisited: number;
  locationEntropy: number;
  sleepDurationHours: number;
  callDurationMinutes: number;
  dailySteps?: number;
  [key: string]: any;
}

export interface BaselineData {
  [featureName: string]: {
    mean: number;
    std: number;
  };
}

// 1. Fetch all registered users
export async function getUsers(): Promise<Patient[]> {
  const usersRef = collection(db, 'users');
  const snap = await getDocs(usersRef);
  return snap.docs.map(doc => ({
    id: doc.id,
    ...doc.data()
  } as Patient));
}

// 2. Fetch latest ML result for a user (useful for admin list sorting)
export async function getLatestResult(uid: string): Promise<MLResult | null> {
  try {
    const resultsRef = collection(db, `users/${uid}/results`);
    const snap = await getDocs(resultsRef);
    if (snap.empty) return null;
    
    // Sort client-side by document ID ("YYYY-MM-DD") descending
    const docs = snap.docs.sort((a, b) => b.id.localeCompare(a.id));
    const d = docs[0];
    return { date: d.id, ...d.data() } as MLResult;
  } catch (err) {
    console.warn(`Could not fetch latest result for ${uid}:`, err);
    return null;
  }
}

// 3. Fetch up to X days of historical results for Line Charts
export async function getHistoricalResults(uid: string, days: number = 14): Promise<MLResult[]> {
  try {
    const resultsRef = collection(db, `users/${uid}/results`);
    const snap = await getDocs(resultsRef);
    if (snap.empty) return [];
    
    const docs = snap.docs.sort((a, b) => b.id.localeCompare(a.id)).slice(0, days);
    return docs.map(d => ({ date: d.id, ...d.data() } as MLResult)).reverse(); // Reverse for chron chart
  } catch (err) {
    console.warn(`Could not fetch history for ${uid}:`, err);
    return [];
  }
}

// 4. Fetch the absolute latest daily_features row (raw sensors)
export async function getLatestFeatures(uid: string): Promise<DailyFeatures | null> {
  try {
    const featsRef = collection(db, `users/${uid}/daily_features`);
    const snap = await getDocs(featsRef);
    if (snap.empty) return null;

    const docs = snap.docs.sort((a, b) => b.id.localeCompare(a.id));
    const d = docs[0];
    return { date: d.id, ...d.data() } as DailyFeatures;
  } catch (err) {
    console.warn(`Could not fetch latest features for ${uid}:`, err);
    return null;
  }
}

// 5. Fetch the established baseline means mapping
export async function getBaseline(uid: string): Promise<BaselineData | null> {
  const baselineRef = collection(db, `users/${uid}/baseline`);
  const snap = await getDocs(baselineRef);
  
  if (snap.empty) return null;

  const baselineMap: BaselineData = {};
  snap.forEach(doc => {
    // Each document is named after the feature, e.g., "screenTimeHours"
    // And contains { mean: X, std: Y }
    baselineMap[doc.id] = doc.data() as { mean: number, std: number };
  });

  return baselineMap;
}
