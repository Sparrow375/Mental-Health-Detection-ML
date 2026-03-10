// App.tsx
import React, {
  useEffect,
  useState,
  createContext,
  useContext,
} from 'react';
import {
  ActivityIndicator,
  View,
  Text,
  StyleSheet,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';

import { loadCheckIns, saveCheckIns } from './src/utils/storage';
import {
  analyzeHistory,
  AnalysisSummary,
  DailyInput,
} from './src/utils/system1';
import DashboardScreen from './src/screens/DashboardScreen';
import CheckInScreen from './src/screens/CheckInScreen';

// --- App-level context ---

type AppContextValue = {
  checkIns: DailyInput[];
  summary: AnalysisSummary;
  addCheckIn: (input: Omit<DailyInput, 'date'>) => Promise<void>;
};

const defaultSummary: AnalysisSummary = {
  baseline: null,
  history: [],
  latest: null,
};

const AppContext = createContext<AppContextValue>({
  checkIns: [],
  summary: defaultSummary,
  addCheckIn: async () => {},
});

export function useAppData() {
  return useContext(AppContext);
}

// --- Tab Navigator ---

const Tab = createBottomTabNavigator();

function TabIcon({
  emoji,
  focused,
}: {
  emoji: string;
  focused: boolean;
}) {
  return (
    <View
      style={[
        tabStyles.iconWrap,
        focused && tabStyles.iconWrapActive,
      ]}
    >
      <Text style={tabStyles.iconEmoji}>{emoji}</Text>
    </View>
  );
}

const tabStyles = StyleSheet.create({
  iconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconWrapActive: {
    backgroundColor: '#EEF2FF',
  },
  iconEmoji: {
    fontSize: 20,
  },
});

// --- Root component ---

export default function App() {
  const [checkIns, setCheckIns] = useState<DailyInput[]>([]);
  const [summary, setSummary] = useState<AnalysisSummary>(defaultSummary);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const stored = await loadCheckIns();
      setCheckIns(stored);
      setSummary(analyzeHistory(stored));
      setLoading(false);
    })();
  }, []);

  const addCheckIn = async (input: Omit<DailyInput, 'date'>) => {
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const newEntry: DailyInput = { date: today, ...input };
    const updated = [
      ...checkIns.filter((d) => d.date !== today),
      newEntry,
    ].sort((a, b) => a.date.localeCompare(b.date));
    setCheckIns(updated);
    setSummary(analyzeHistory(updated));
    await saveCheckIns(updated);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.loadingLogo}>🧠</Text>
        <Text style={styles.loadingTitle}>MoodGuard</Text>
        <ActivityIndicator
          size="large"
          color="#6366F1"
          style={{ marginTop: 24 }}
        />
        <Text style={styles.loadingText}>Loading your data…</Text>
      </View>
    );
  }

  return (
    <AppContext.Provider value={{ checkIns, summary, addCheckIn }}>
      <StatusBar style="dark" />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={{
            headerShown: false,
            tabBarStyle: {
              backgroundColor: '#fff',
              borderTopColor: '#F3F4F6',
              borderTopWidth: 1,
              height: 64,
              paddingBottom: 8,
              paddingTop: 4,
            },
            tabBarActiveTintColor: '#6366F1',
            tabBarInactiveTintColor: '#9CA3AF',
            tabBarLabelStyle: {
              fontSize: 12,
              fontWeight: '600',
            },
          }}
        >
          <Tab.Screen
            name="Dashboard"
            component={DashboardScreen}
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon emoji="📊" focused={focused} />
              ),
            }}
          />
          <Tab.Screen
            name="Check-in"
            component={CheckInScreen}
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon emoji="✏️" focused={focused} />
              ),
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </AppContext.Provider>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F7F8FC',
  },
  loadingLogo: {
    fontSize: 56,
    marginBottom: 8,
  },
  loadingTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1A1A2E',
    letterSpacing: -0.5,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#9CA3AF',
  },
});
