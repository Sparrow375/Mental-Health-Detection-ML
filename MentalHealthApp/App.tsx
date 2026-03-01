import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Text } from 'react-native';

import HomeScreen from './src/screens/HomeScreen';
import LogDataScreen from './src/screens/LogDataScreen';
import ReportsScreen from './src/screens/ReportsScreen';
import MetricsScreen from './src/screens/MetricsScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import { Colors } from './src/theme';
import { requestPermissions, scheduleDailyReminder } from './src/services/notifications';
import { getNotifyTime, isSetupDone, markSetupDone, saveBaseline } from './src/services/storage';
import { generateSyntheticBaseline } from './src/engine/baselineGenerator';

const Tab = createBottomTabNavigator();

export default function App() {
  useEffect(() => {
    (async () => {
      const done = await isSetupDone();
      if (!done) { await saveBaseline(generateSyntheticBaseline()); await markSetupDone(); }
      const granted = await requestPermissions();
      if (granted) { const time = await getNotifyTime(); await scheduleDailyReminder(time); }
    })();
  }, []);

  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={{
          headerStyle: { backgroundColor: Colors.surface },
          headerTintColor: Colors.text,
          headerTitleStyle: { fontWeight: '700' },
          tabBarStyle: { backgroundColor: Colors.surface, borderTopColor: Colors.border, borderTopWidth: 1, height: 60, paddingBottom: 8 },
          tabBarActiveTintColor: Colors.primary,
          tabBarInactiveTintColor: Colors.textMuted,
          tabBarLabelStyle: { fontSize: 11, fontWeight: '600' },
        }}>
        <Tab.Screen name="Home" component={HomeScreen}
          options={{ headerTitle: '🧠 MH Monitor', tabBarLabel: 'Home', tabBarIcon: ({ focused }) => <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>🏠</Text> }} />
        <Tab.Screen name="LogData" component={LogDataScreen}
          options={{ headerTitle: 'Log Data', tabBarLabel: 'Log', tabBarIcon: ({ focused }) => <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>✏️</Text> }} />
        <Tab.Screen name="Reports" component={ReportsScreen}
          options={{ headerTitle: 'Reports', tabBarLabel: 'Reports', tabBarIcon: ({ focused }) => <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>📋</Text> }} />
        <Tab.Screen name="Metrics" component={MetricsScreen}
          options={{ headerTitle: 'Metrics', tabBarLabel: 'Metrics', tabBarIcon: ({ focused }) => <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>📊</Text> }} />
        <Tab.Screen name="Settings" component={SettingsScreen}
          options={{ headerTitle: 'Settings', tabBarLabel: 'Settings', tabBarIcon: ({ focused }) => <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>⚙️</Text> }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
