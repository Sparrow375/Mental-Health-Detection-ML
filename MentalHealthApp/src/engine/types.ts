// ============================================================================
// DATA STRUCTURES — ported from system1.py
// ============================================================================

export interface PersonalityVector {
    screen_time_hours: number;
    unlock_count: number;
    social_app_ratio: number;
    calls_per_day: number;
    texts_per_day: number;
    unique_contacts: number;
    response_time_minutes: number;
    daily_displacement_km: number;
    location_entropy: number;
    home_time_ratio: number;
    places_visited: number;
    wake_time_hour: number;
    sleep_time_hour: number;
    sleep_duration_hours: number;
    dark_duration_hours: number;
    charge_duration_hours: number;
    conversation_duration_hours: number;
    conversation_frequency: number;
    variances: Record<string, number>;
}

export const FEATURE_KEYS: (keyof Omit<PersonalityVector, 'variances'>)[] = [
    'screen_time_hours', 'unlock_count', 'social_app_ratio', 'calls_per_day',
    'texts_per_day', 'unique_contacts', 'response_time_minutes',
    'daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited',
    'wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours',
    'dark_duration_hours', 'charge_duration_hours',
    'conversation_duration_hours', 'conversation_frequency',
];

export const FEATURE_LABELS: Record<string, string> = {
    screen_time_hours: 'Screen Time (hrs)',
    unlock_count: 'Phone Unlocks',
    social_app_ratio: 'Social App Ratio',
    calls_per_day: 'Calls / Day',
    texts_per_day: 'Texts / Day',
    unique_contacts: 'Unique Contacts',
    response_time_minutes: 'Response Time (min)',
    daily_displacement_km: 'Daily Movement (km)',
    location_entropy: 'Location Entropy',
    home_time_ratio: 'Home Time Ratio',
    places_visited: 'Places Visited',
    wake_time_hour: 'Wake Time (24h)',
    sleep_time_hour: 'Sleep Time (24h)',
    sleep_duration_hours: 'Sleep Duration (hrs)',
    dark_duration_hours: 'Dark Duration (hrs)',
    charge_duration_hours: 'Charge Duration (hrs)',
    conversation_duration_hours: 'Conversation Duration (hrs)',
    conversation_frequency: 'Conversation Frequency',
};

export const FEATURE_DEFAULTS: Omit<PersonalityVector, 'variances'> = {
    screen_time_hours: 4.5, unlock_count: 80, social_app_ratio: 0.35,
    calls_per_day: 3, texts_per_day: 25, unique_contacts: 8,
    response_time_minutes: 15, daily_displacement_km: 12,
    location_entropy: 2.3, home_time_ratio: 0.65, places_visited: 4,
    wake_time_hour: 7.5, sleep_time_hour: 23.5, sleep_duration_hours: 7.5,
    dark_duration_hours: 8.5, charge_duration_hours: 6,
    conversation_duration_hours: 1.5, conversation_frequency: 12,
};

export const FEATURE_RANGES: Record<string, { min: number; max: number }> = {
    screen_time_hours: { min: 0, max: 18 },
    unlock_count: { min: 0, max: 300 },
    social_app_ratio: { min: 0, max: 1 },
    calls_per_day: { min: 0, max: 50 },
    texts_per_day: { min: 0, max: 500 },
    unique_contacts: { min: 0, max: 100 },
    response_time_minutes: { min: 0, max: 600 },
    daily_displacement_km: { min: 0, max: 100 },
    location_entropy: { min: 0, max: 5 },
    home_time_ratio: { min: 0, max: 1 },
    places_visited: { min: 0, max: 30 },
    wake_time_hour: { min: 0, max: 23 },
    sleep_time_hour: { min: 0, max: 23 },
    sleep_duration_hours: { min: 0, max: 16 },
    dark_duration_hours: { min: 0, max: 24 },
    charge_duration_hours: { min: 0, max: 24 },
    conversation_duration_hours: { min: 0, max: 10 },
    conversation_frequency: { min: 0, max: 100 },
};

export type AlertLevel = 'green' | 'yellow' | 'orange' | 'red';
export type PatternType =
    | 'stable' | 'rapid_cycling' | 'gradual_drift'
    | 'acute_spike' | 'mixed_pattern' | 'insufficient_data';

export interface DailyReport {
    dayNumber: number;
    date: string;
    anomalyScore: number;
    alertLevel: AlertLevel;
    flaggedFeatures: string[];
    patternType: PatternType;
    sustainedDeviationDays: number;
    evidenceAccumulated: number;
    topDeviations: Record<string, number>;
    notes: string;
    featuresEntered: Partial<Omit<PersonalityVector, 'variances'>>;
}

export interface AnomalyReport {
    timestamp: string;
    overallAnomalyScore: number;
    featureDeviations: Record<string, number>;
    deviationVelocity: Record<string, number>;
    alertLevel: AlertLevel;
    flaggedFeatures: string[];
    patternType: PatternType;
    sustainedDeviationDays: number;
    evidenceAccumulated: number;
}

export interface FinalPrediction {
    patientId: string;
    monitoringDays: number;
    finalAnomalyScore: number;
    sustainedAnomalyDetected: boolean;
    confidence: number;
    patternIdentified: string;
    evidenceSummary: {
        sustainedDeviationDays: number;
        maxSustainedDays: number;
        evidenceAccumulatedFinal: number;
        peakEvidence: number;
        maxDailyAnomalyScore: number;
        avgRecentAnomalyScore: number;
        monitoringDays: number;
        daysAboveThreshold: number;
    };
    recommendation: string;
}

export interface DetectorState {
    sustainedDeviationDays: number;
    evidenceAccumulated: number;
    maxEvidence: number;
    maxSustainedDays: number;
    maxAnomalyScore: number;
    anomalyScoreHistory: number[];
    featureHistory: Record<string, number[]>;
}
