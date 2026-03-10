// src/screens/DashboardScreen.tsx
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import { useAppData } from '../../App';
import type { DailyAnalysis } from '../utils/system1';

function alertColor(level: DailyAnalysis['alertLevel']): string {
  switch (level) {
    case 'green':
      return '#4CAF50';
    case 'yellow':
      return '#FFC107';
    case 'orange':
      return '#FF9800';
    case 'red':
      return '#F44336';
    default:
      return '#4CAF50';
  }
}

function alertBg(level: DailyAnalysis['alertLevel']): string {
  switch (level) {
    case 'green':
      return '#E8F5E9';
    case 'yellow':
      return '#FFFDE7';
    case 'orange':
      return '#FFF3E0';
    case 'red':
      return '#FFEBEE';
    default:
      return '#E8F5E9';
  }
}

function alertMessage(level: DailyAnalysis['alertLevel']): string {
  switch (level) {
    case 'green':
      return 'Your patterns look normal. Keep it up!';
    case 'yellow':
      return 'Minor deviations detected. Keep an eye on your routine.';
    case 'orange':
      return 'Notable changes detected. Consider talking to someone.';
    case 'red':
      return 'Significant deviations sustained. Please reach out for support.';
    default:
      return '';
  }
}

export default function DashboardScreen() {
  const { summary, checkIns } = useAppData();
  const latest = summary.latest;

  const daysUntilBaseline = Math.max(0, 14 - checkIns.length);
  const baselineProgress = Math.min(checkIns.length / 14, 1);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.header}>MoodGuard</Text>
        <Text style={styles.subheader}>Mental Health System 1 Monitor</Text>

        {/* Alert Status Card */}
        {latest ? (
          <View
            style={[
              styles.alertCard,
              {
                borderColor: alertColor(latest.alertLevel),
                backgroundColor: alertBg(latest.alertLevel),
              },
            ]}
          >
            <View style={styles.alertRow}>
              <View
                style={[
                  styles.alertBadge,
                  { backgroundColor: alertColor(latest.alertLevel) },
                ]}
              >
                <Text style={styles.alertBadgeText}>
                  {latest.alertLevel.toUpperCase()}
                </Text>
              </View>
              <Text style={styles.alertScore}>
                Score: {latest.anomalyScore.toFixed(3)}
              </Text>
            </View>
            <Text style={styles.alertMessage}>
              {alertMessage(latest.alertLevel)}
            </Text>
            <View style={styles.statsRow}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {latest.sustainedDeviationDays}d
                </Text>
                <Text style={styles.statLabel}>Sustained</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {latest.evidenceAccumulated.toFixed(2)}
                </Text>
                <Text style={styles.statLabel}>Evidence</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{checkIns.length}</Text>
                <Text style={styles.statLabel}>Days logged</Text>
              </View>
            </View>
          </View>
        ) : (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyIcon}>📋</Text>
            <Text style={styles.emptyTitle}>No data yet</Text>
            <Text style={styles.emptyText}>
              Complete your first daily check-in to start monitoring.
            </Text>
          </View>
        )}

        {/* Baseline progress */}
        {!summary.baseline && checkIns.length > 0 && (
          <View style={styles.progressCard}>
            <Text style={styles.progressLabel}>
              Building your baseline: {checkIns.length}/14 days
            </Text>
            <View style={styles.progressTrack}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${baselineProgress * 100}%` },
                ]}
              />
            </View>
            <Text style={styles.progressSub}>
              {daysUntilBaseline} more check-in
              {daysUntilBaseline !== 1 ? 's' : ''} needed for anomaly detection
            </Text>
          </View>
        )}

        {/* History list */}
        <Text style={styles.sectionTitle}>Last 14 Days</Text>
        {summary.history.length === 0 ? (
          <Text style={styles.emptyListText}>
            Your history will appear here after your first check-in.
          </Text>
        ) : (
          <FlatList
            data={[...summary.history].slice(-14).reverse()}
            keyExtractor={(item) => item.date}
            scrollEnabled={false}
            renderItem={({ item }) => (
              <View style={styles.row}>
                <View style={styles.rowLeft}>
                  <View
                    style={[
                      styles.dot,
                      { backgroundColor: alertColor(item.alertLevel) },
                    ]}
                  />
                  <View>
                    <Text style={styles.rowDate}>{item.date}</Text>
                    <Text style={styles.rowLevel}>
                      {item.alertLevel.charAt(0).toUpperCase() +
                        item.alertLevel.slice(1)}
                    </Text>
                  </View>
                </View>
                <View style={styles.rowRight}>
                  <Text style={styles.rowScore}>
                    {item.anomalyScore.toFixed(3)}
                  </Text>
                  <Text style={styles.rowScoreLabel}>score</Text>
                </View>
              </View>
            )}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F7F8FC',
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
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
    marginBottom: 20,
    marginTop: 2,
  },
  alertCard: {
    borderWidth: 2,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  alertRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  alertBadge: {
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  alertBadgeText: {
    color: '#fff',
    fontWeight: '800',
    fontSize: 16,
    letterSpacing: 1,
  },
  alertScore: {
    fontSize: 14,
    color: '#374151',
    fontWeight: '600',
  },
  alertMessage: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
    marginBottom: 16,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statValue: {
    fontSize: 22,
    fontWeight: '800',
    color: '#1A1A2E',
  },
  statLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginTop: 2,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  statDivider: {
    width: 1,
    backgroundColor: '#E5E7EB',
    marginVertical: 4,
  },
  emptyCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  emptyIcon: {
    fontSize: 40,
    marginBottom: 12,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1A1A2E',
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 20,
  },
  progressCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  progressLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 10,
  },
  progressTrack: {
    height: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#6366F1',
    borderRadius: 4,
  },
  progressSub: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1A1A2E',
    marginBottom: 12,
    marginTop: 8,
  },
  emptyListText: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    marginTop: 16,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 8,
    shadowColor: '#000',
    shadowOpacity: 0.03,
    shadowRadius: 4,
    elevation: 1,
  },
  rowLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  rowDate: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1A1A2E',
  },
  rowLevel: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 1,
  },
  rowRight: {
    alignItems: 'flex-end',
  },
  rowScore: {
    fontSize: 16,
    fontWeight: '700',
    color: '#374151',
    fontVariant: ['tabular-nums'],
  },
  rowScoreLabel: {
    fontSize: 11,
    color: '#9CA3AF',
    textTransform: 'uppercase',
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
});
