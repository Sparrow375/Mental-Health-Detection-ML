export const Colors = {
    bg: '#0f1117', surface: '#1c1f2e', card: '#252838', border: '#2e3147',
    text: '#e8eaf6', textMuted: '#7c7f9e', primary: '#6c63ff', primaryLight: '#9d96ff',
    green: '#4caf50', greenDim: '#1b3a1c', yellow: '#ffd740', yellowDim: '#3a3000',
    orange: '#ff9100', orangeDim: '#3a1f00', red: '#f44336', redDim: '#3a0a08',
};
export const ALERT_COLORS: Record<string, string> = {
    green: Colors.green, yellow: Colors.yellow, orange: Colors.orange, red: Colors.red,
};
export const ALERT_BG: Record<string, string> = {
    green: Colors.greenDim, yellow: Colors.yellowDim, orange: Colors.orangeDim, red: Colors.redDim,
};
export const ALERT_LABELS: Record<string, string> = {
    green: '✅ Normal', yellow: '⚠️ Watch', orange: '🟠 Concern', red: '🚨 Alert',
};
