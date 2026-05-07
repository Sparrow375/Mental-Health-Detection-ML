package com.example.mhealth.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.ui.theme.*
import com.example.mhealth.ui.charts.SparklineChart

@Composable
fun ScreenHeader(
    title: String,
    subtitle: String,
    icon: ImageVector,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 20.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(OceanBlue.copy(alpha = 0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, contentDescription = null, tint = OceanBlue, modifier = Modifier.size(24.dp))
        }
        Spacer(Modifier.width(16.dp))
        Column {
            Text(title, fontSize = 20.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            Text(subtitle, fontSize = 12.sp, color = TextSecondary)
        }
    }
}

@Composable
fun InfoCard(
    title: String,
    headerColor: Color = OceanBlue,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .shadow(4.dp, RoundedCornerShape(20.dp)),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = CardWhite),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(bottom = 12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .width(4.dp)
                        .height(18.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(headerColor)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    title,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
            }
            content()
        }
    }
}

@Composable
fun MetricPill(
    label: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.padding(4.dp)
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(14.dp))
                .background(
                    Brush.verticalGradient(
                        listOf(color.copy(alpha = 0.15f), color.copy(alpha = 0.08f))
                    )
                )
                .border(1.dp, color.copy(alpha = 0.2f), RoundedCornerShape(14.dp))
                .padding(horizontal = 14.dp, vertical = 8.dp)
        ) {
            Text(
                value,
                fontSize = 15.sp,
                fontWeight = FontWeight.ExtraBold,
                color = color
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            label,
            fontSize = 10.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun SparklineLabel(
    label: String,
    history: List<Float>,
    color: Color,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(label, fontSize = 12.sp, color = TextSecondary)
            if (history.isNotEmpty()) {
                val current = history.last()
                Text(
                    if (current % 1 == 0f) "${current.toInt()}" else "%.1f".format(current),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
            }
        }
        SparklineChart(
            values = history,
            color = color,
            modifier = Modifier
                .width(80.dp)
                .height(30.dp)
        )
    }
}

@Composable
fun ComparisonRow(
    label: String,
    current: Float,
    baseline: Float,
    modifier: Modifier = Modifier
) {
    val diff = current - baseline
    val percent = if (baseline != 0f) (diff / baseline) * 100 else 0f

    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, fontSize = 13.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
            Text(
                "Current: ${"%.1f".format(current)} (Base: ${"%.1f".format(baseline)})",
                fontSize = 11.sp,
                color = TextMuted
            )
        }

        val statusColor = when {
            percent > 20f -> AlertRed
            percent < -20f -> AlertGreen
            else -> TextSecondary
        }

        Text(
            text = if (percent >= 0) "+${"%.0f".format(percent)}%" else "${"%.0f".format(percent)}%",
            color = statusColor,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun CollapsibleCard(
    title: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    expanded: Boolean,
    onToggle: () -> Unit,
    content: @Composable () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(3.dp, RoundedCornerShape(16.dp)),
        colors = CardDefaults.cardColors(containerColor = CardLight),
        shape = RoundedCornerShape(16.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column {
            // Accent bar on left
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onToggle() }
                    .padding(start = 4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .width(3.dp)
                        .height(60.dp)
                        .padding(vertical = 16.dp)
                        .background(AccentBlue, RoundedCornerShape(2.dp))
                )
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(icon, null, tint = AccentBlue, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                        Text(subtitle, color = TextSecondary, fontSize = 11.sp)
                    }
                    Icon(
                        if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        null, tint = TextSecondary, modifier = Modifier.size(20.dp)
                    )
                }
            }

            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = fadeOut()
            ) {
                Column(modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
                    HorizontalDivider(color = BorderLight)
                    Spacer(Modifier.height(8.dp))
                    content()
                }
            }
        }
    }
}

@Composable
fun MiniStat(label: String, value: String, color: Color) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.padding(horizontal = 2.dp)
    ) {
        Text(value, color = color, fontSize = 11.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(1.dp))
        Box(modifier = Modifier.width(20.dp).height(2.dp).background(color.copy(alpha = 0.4f), RoundedCornerShape(1.dp)))
        Spacer(Modifier.height(2.dp))
        Text(label, color = TextSecondary, fontSize = 9.sp)
    }
}

@Composable
fun PhoneMetric(label: String, value: String, color: Color) {
    Card(
        modifier = Modifier.padding(2.dp),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.06f)),
        shape = RoundedCornerShape(8.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp)
        ) {
            Text(value, color = color, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Text(label, color = TextSecondary, fontSize = 9.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
    }
}

@Composable
fun TextureMetric(label: String, value: String, color: Color) {
    Card(
        modifier = Modifier.padding(2.dp),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.06f)),
        shape = RoundedCornerShape(8.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
        ) {
            Text(value, color = color, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            Text(label, color = TextSecondary, fontSize = 8.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
    }
}
