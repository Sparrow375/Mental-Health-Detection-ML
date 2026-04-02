package com.example.mhealth.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.ui.components.*
import com.example.mhealth.ui.charts.ArcProgressRing
import com.example.mhealth.ui.theme.*

@Composable
fun MonitorScreen() {
    val vector by DataRepository.latestVector.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()

    LazyColumn(Modifier.fillMaxSize()) {
        item { ScreenHeader("Advanced Sensors", "Real-time behavioral stream analysis", Icons.Default.Sensors) }
        
        // 1. Primary Metrics (Arcs)
        item {
            InfoCard("Core Activity Window", headerColor = MhealthIndigo) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                    vector?.let { v ->
                        ArcProgressRing(v.screenTimeHours, 12f, MhealthIndigo, "Screen", "hrs")
                        ArcProgressRing(v.sleepDurationHours, 12f, MhealthTeal, "Sleep", "hrs")
                        ArcProgressRing(v.socialAppRatio * 100f, 100f, MhealthAccentPurple, "Social", "%")
                    }
                }
            }
        }

        // 2. Behavioral Pills
        item {
            InfoCard("Behavioral Statistics", headerColor = MhealthTeal) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                    vector?.let { v ->
                        MetricPill("Unlocks", "${v.unlockCount}", MhealthIndigo)
                        MetricPill("Places", "${v.placesVisited}", MhealthTeal)
                        MetricPill("Displac.", "%.1fkm".format(v.dailyDisplacementKm), MhealthChartCoral)
                    }
                }
            }
        }

        // 3. Trends (Sparklines)
        item {
            val history by DataRepository.weeklyFeatureHistory.collectAsState(initial = emptyList())
            InfoCard("Weekly Trends", headerColor = MhealthChartIndigo) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SparklineLabel("Screen Time (7d)", history.map { it.screenTimeHours }, MhealthIndigo)
                    SparklineLabel("Sleep Quality (7d)", history.map { it.sleepDurationHours }, MhealthTeal)
                    SparklineLabel("Movement Entropy (7d)", history.map { it.locationEntropy }, MhealthChartCoral)
                }
            }
        }

        // 4. Comparison vs Baseline
        if (baseline != null && vector != null) {
            item {
                InfoCard("Deviation vs Baseline", headerColor = MhealthChartSlate) {
                    val b = checkNotNull(baseline)
                    val v = checkNotNull(vector)
                    Column {
                        ComparisonRow("Screen Time", v.screenTimeHours, b.screenTimeHours)
                        ComparisonRow("Social Usage", v.socialAppRatio, b.socialAppRatio)
                        ComparisonRow("Location Diversity", v.locationEntropy, b.locationEntropy)
                    }
                }
            }
        }

        item { Spacer(Modifier.height(12.dp)) }
    }
}
