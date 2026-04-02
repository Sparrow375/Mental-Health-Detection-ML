import { collection, query, orderBy, limit, getDocs, doc, getDoc, where, deleteDoc, writeBatch } from 'firebase/firestore';
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
  all_scores_json?: string;       // JSON blob of all prototype match scores (from CloudSyncWorker)
  prototype_confidence?: number;   // 0.0–1.0 confidence of top match
  alert_level?: string;           // "green" | "yellow" | "orange" | "red"
  [key: string]: any;
}

export interface DailyFeatures {
  date: string;
  screenTimeHours?: number;
  unlockCount?: number;
  appLaunchCount?: number;
  notificationsToday?: number;
  socialAppRatio?: number;
  callsPerDay?: number;
  callDurationMinutes?: number;
  uniqueContacts?: number;
  conversationFrequency?: number;
  dailyDisplacementKm?: number;
  locationEntropy?: number;
  homeTimeRatio?: number;
  placesVisited?: number;
  wakeTimeHour?: number;
  sleepTimeHour?: number;
  sleepDurationHours?: number;
  darkDurationHours?: number;
  chargeDurationHours?: number;
  memoryUsagePercent?: number;
  networkWifiMB?: number;
  networkMobileMB?: number;
  downloadsToday?: number;
  storageUsedGB?: number;
  appUninstallsToday?: number;
  upiTransactionsToday?: number;
  totalAppsCount?: number;
  dailySteps?: number;
  appBreakdown?: Record<string, any>;
  notificationBreakdown?: Record<string, any>;
  appLaunchesBreakdown?: Record<string, any>;
  [key: string]: any;
}

export interface BaselineData {
  [featureName: string]: {
    mean: number;
    std: number;
    baselineValue?: number;
    stdDeviation?: number;
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
    const q = query(resultsRef, orderBy('__name__', 'desc'), limit(1));
    const snap = await getDocs(q);
    if (snap.empty) return null;
    
    const d = snap.docs[0];
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
    const q = query(resultsRef, orderBy('__name__', 'desc'), limit(days));
    const snap = await getDocs(q);
    if (snap.empty) return [];
    
    return snap.docs.map(d => ({ date: d.id, ...d.data() } as MLResult)).reverse(); // Reverse for chron chart
  } catch (err) {
    console.warn(`Could not fetch history for ${uid}:`, err);
    return [];
  }
}

// Helper: parse Firestore daily_features doc into DailyFeatures
function parseDailyDoc(docId: string, data: Record<string, any>): DailyFeatures {
  const parsed: DailyFeatures = { date: docId, ...data };
  // The Android/System2 backend stores breakdown fields as JSON strings with 'Json' suffix
  const jsonFields: [string, string][] = [
    ['appBreakdownJson', 'appBreakdown'],
    ['notificationBreakdownJson', 'notificationBreakdown'],
    ['appLaunchesBreakdownJson', 'appLaunchesBreakdown'],
  ];
  for (const [jsonKey, objKey] of jsonFields) {
    if (typeof data[jsonKey] === 'string') {
      try { parsed[objKey] = JSON.parse(data[jsonKey]); } catch { parsed[objKey] = {}; }
    }
  }
  return parsed;
}

// 4. Fetch the absolute latest daily_features row (raw sensors)
// Checks both 'daily_features' (System2 backend) and 'daily_data' (CloudSyncWorker) collections
export async function getLatestFeatures(uid: string): Promise<DailyFeatures | null> {
  try {
    // Try daily_features first (where System2 backend writes)
    let featsRef = collection(db, `users/${uid}/daily_features`);
    let q = query(featsRef, orderBy('__name__', 'desc'), limit(1));
    let snap = await getDocs(q);

    // Fallback to daily_data (where CloudSyncWorker writes)
    if (snap.empty) {
      featsRef = collection(db, `users/${uid}/daily_data`);
      q = query(featsRef, orderBy('__name__', 'desc'), limit(1));
      snap = await getDocs(q);
    }

    if (snap.empty) return null;

    const d = snap.docs[0];
    return parseDailyDoc(d.id, d.data());
  } catch (err) {
    console.warn(`Could not fetch latest features for ${uid}:`, err);
    return null;
  }
}

// 4b. Fetch ALL daily feature records (for day-by-day navigation)
export async function getAllDailyFeatures(uid: string): Promise<DailyFeatures[]> {
  try {
    let featsRef = collection(db, `users/${uid}/daily_features`);
    let snap = await getDocs(featsRef);

    if (snap.empty) {
      featsRef = collection(db, `users/${uid}/daily_data`);
      snap = await getDocs(featsRef);
    }

    if (snap.empty) return [];

    return snap.docs
      .sort((a, b) => b.id.localeCompare(a.id)) // newest first
      .map(d => parseDailyDoc(d.id, d.data()));
  } catch (err) {
    console.warn(`Could not fetch all daily features for ${uid}:`, err);
    return [];
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
    // And contains { baselineValue: X, stdDeviation: Y }
    const d = doc.data();
    baselineMap[doc.id] = {
      mean: typeof d.baselineValue === 'number' ? d.baselineValue : (d.mean || 0),
      std: typeof d.stdDeviation === 'number' ? d.stdDeviation : (d.std || 1)
    };
  });

  return baselineMap;
}

// 6. Delete Patient and All Subcollections
export async function deletePatient(uid: string): Promise<boolean> {
  try {
    const batch = writeBatch(db);
    
    // Helper to delete all documents in a subcollection
    const deleteSubcollection = async (subName: string) => {
      const snap = await getDocs(collection(db, `users/${uid}/${subName}`));
      snap.forEach(d => batch.delete(d.ref));
    };

    await deleteSubcollection('daily_features');
    await deleteSubcollection('daily_data');
    await deleteSubcollection('results');
    await deleteSubcollection('baseline');
    await deleteSubcollection('location_data');

    // Delete the root user document
    batch.delete(doc(db, 'users', uid));

    await batch.commit();
    return true;
  } catch (err) {
    console.error(`Failed to delete patient ${uid}:`, err);
    return false;
  }
}
