package com.example.mhealth.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.models.PersonalityVector
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.cos
import kotlin.math.sin

val DimensionRestColor = Color(0xFF38BDF8)
val DimensionSocialColor = Color(0xFFF472B6)
val DimensionMovementColor = Color(0xFF34D399)
val DimensionDigitalColor = Color(0xFFA78BFA)

@Composable
fun RhythmTrendsChart(
    features: List<PersonalityVector>,
    baseline: PersonalityVector?,
    selectedDaysCount: Int = 14,
    onDaySelected: (PersonalityVector) -> Unit = {}
) {
    if (features.isEmpty()) return

    val displayList = remember(features, selectedDaysCount) {
        features.takeLast(selectedDaysCount)
    }

    val dateLabels = remember(displayList) {
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -(displayList.size - 1))
        displayList.map {
            val label = SimpleDateFormat("MMM d", Locale.getDefault()).format(cal.time)
            cal.add(Calendar.DAY_OF_YEAR, 1)
            label
        }
    }

    val textMeasurer = rememberTextMeasurer()
    val onSurfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Rhythm Trends",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "% of Personal Baseline",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    LegendItem("Rest", DimensionRestColor)
                    LegendItem("Social", DimensionSocialColor)
                    LegendItem("Movement", DimensionMovementColor)
                    LegendItem("Digital", DimensionDigitalColor)
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val width = size.width
                    val height = size.height
                    val paddingLeft = 40.dp.toPx()
                    val paddingBottom = 24.dp.toPx()
                    val graphWidth = width - paddingLeft
                    val graphHeight = height - paddingBottom

                    // Draw horizontal reference lines (50%, 100%, 150%)
                    val levels = listOf(0.5f, 1.0f, 1.5f)
                    levels.forEach { level ->
                        val y = graphHeight * (1.0f - (level - 0.3f) / 1.4f)
                        val stroke = if (level == 1.0f) {
                            Stroke(width = 1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f), 0f))
                        } else {
                            Stroke(width = 0.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(5f, 5f), 0f))
                        }
                        drawLine(
                            color = onSurfaceVariant.copy(alpha = if (level == 1.0f) 0.4f else 0.2f),
                            start = Offset(paddingLeft, y),
                            end = Offset(width, y),
                            strokeWidth = stroke.width,
                            pathEffect = stroke.pathEffect
                        )

                        val pctText = "${(level * 100).toInt()}%"
                        drawText(
                            textMeasurer = textMeasurer,
                            text = pctText,
                            style = TextStyle(fontSize = 9.sp, color = onSurfaceVariant.copy(0.6f)),
                            topLeft = Offset(4.dp.toPx(), y - 8.dp.toPx())
                        )
                    }

                    if (displayList.size > 1) {
                        val baseRest = baseline?.sleepDurationHours?.coerceAtLeast(1.0f) ?: 7.0f
                        val baseSocial = baseline?.callsPerDay?.coerceAtLeast(1.0f) ?: 3.0f
                        val baseMove = baseline?.dailyStepCount?.coerceAtLeast(500.0f) ?: 2500.0f
                        val baseDigital = baseline?.screenTimeHours?.coerceAtLeast(1.0f) ?: 4.0f

                        val count = displayList.size
                        val stepX = graphWidth / (count - 1).coerceAtLeast(1)

                        fun calcY(ratio: Float): Float {
                            val clamped = ratio.coerceIn(0.3f, 1.7f)
                            return graphHeight * (1.0f - (clamped - 0.3f) / 1.4f)
                        }

                        // Draw lines for each dimension
                        val dims = listOf(
                            DimensionRestColor to displayList.map { it.sleepDurationHours / baseRest },
                            DimensionSocialColor to displayList.map { it.callsPerDay / baseSocial },
                            DimensionMovementColor to displayList.map { it.dailyStepCount / baseMove },
                            DimensionDigitalColor to displayList.map { it.screenTimeHours / baseDigital }
                        )

                        dims.forEach { (color, series) ->
                            val path = Path()
                            series.forEachIndexed { i, ratio ->
                                val x = paddingLeft + i * stepX
                                val y = calcY(ratio)
                                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                            }
                            drawPath(
                                path = path,
                                color = color,
                                style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round)
                            )
                        }

                        // Draw X-axis date labels
                        val labelInterval = (count / 4).coerceAtLeast(1)
                        displayList.forEachIndexed { i, _ ->
                            if (i % labelInterval == 0 || i == count - 1) {
                                val x = paddingLeft + i * stepX
                                val label = dateLabels.getOrNull(i) ?: ""
                                drawText(
                                    textMeasurer = textMeasurer,
                                    text = label,
                                    style = TextStyle(fontSize = 9.sp, color = onSurfaceVariant.copy(0.6f)),
                                    topLeft = Offset(x - 15.dp.toPx(), graphHeight + 4.dp.toPx())
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun LegendItem(label: String, color: Color) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(color, CircleShape)
        )
        Text(
            text = label,
            fontSize = 10.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
fun BehavioralFingerprintRadar(
    currentVector: PersonalityVector?,
    baselineVector: PersonalityVector?
) {
    val labels = listOf("Rest", "Social", "Movement", "Digital", "Cadence", "Daylight")
    val textMeasurer = rememberTextMeasurer()
    val primaryColor = MaterialTheme.colorScheme.primary
    val outlineColor = MaterialTheme.colorScheme.outline
    val onSurfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant

    // Normalize ratios relative to baseline
    val currentRatios = remember(currentVector, baselineVector) {
        if (currentVector == null || baselineVector == null) {
            listOf(0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f)
        } else {
            val baseRest = baselineVector.sleepDurationHours.coerceAtLeast(1.0f)
            val baseSocial = baselineVector.callsPerDay.coerceAtLeast(1.0f)
            val baseMove = baselineVector.dailyStepCount.coerceAtLeast(500.0f)
            val baseDigital = baselineVector.screenTimeHours.coerceAtLeast(1.0f)
            val baseCadence = baselineVector.keystrokeSpeed.coerceAtLeast(1.0f)
            val baseDaylight = baselineVector.daylightExposureMinutes.coerceAtLeast(10.0f)

            listOf(
                (currentVector.sleepDurationHours / baseRest).coerceIn(0.2f, 1.5f),
                (currentVector.callsPerDay / baseSocial).coerceIn(0.2f, 1.5f),
                (currentVector.dailyStepCount / baseMove).coerceIn(0.2f, 1.5f),
                (currentVector.screenTimeHours / baseDigital).coerceIn(0.2f, 1.5f),
                (currentVector.keystrokeSpeed / baseCadence).coerceIn(0.2f, 1.5f),
                (currentVector.daylightExposureMinutes / baseDaylight).coerceIn(0.2f, 1.5f)
            )
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Behavioral Fingerprint",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "This Week vs Baseline",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    LegendItem("Baseline", outlineColor.copy(0.6f))
                    LegendItem("This Week", primaryColor)
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val centerX = size.width / 2f
                    val centerY = size.height / 2f
                    val radius = (size.height / 2f) - 30.dp.toPx()
                    val numAxes = 6

                    // Draw concentric webs (25%, 50%, 75%, 100%)
                    for (level in 1..4) {
                        val r = radius * (level / 4f)
                        val path = Path()
                        for (i in 0 until numAxes) {
                            val angle = Math.toRadians((i * 60 - 90).toDouble())
                            val x = centerX + (r * cos(angle)).toFloat()
                            val y = centerY + (r * sin(angle)).toFloat()
                            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                        }
                        path.close()
                        drawPath(
                            path = path,
                            color = outlineColor.copy(alpha = 0.15f),
                            style = Stroke(width = 1.dp.toPx())
                        )
                    }

                    // Draw spider web spokes & axis labels
                    for (i in 0 until numAxes) {
                        val angle = Math.toRadians((i * 60 - 90).toDouble())
                        val endX = centerX + (radius * cos(angle)).toFloat()
                        val endY = centerY + (radius * sin(angle)).toFloat()
                        drawLine(
                            color = outlineColor.copy(alpha = 0.2f),
                            start = Offset(centerX, centerY),
                            end = Offset(endX, endY),
                            strokeWidth = 1.dp.toPx()
                        )

                        // Label text position
                        val labelX = centerX + ((radius + 18.dp.toPx()) * cos(angle)).toFloat() - 15.dp.toPx()
                        val labelY = centerY + ((radius + 18.dp.toPx()) * sin(angle)).toFloat() - 8.dp.toPx()
                        drawText(
                            textMeasurer = textMeasurer,
                            text = labels[i],
                            style = TextStyle(fontSize = 10.sp, color = onSurfaceVariant),
                            topLeft = Offset(labelX, labelY)
                        )
                    }

                    // Draw Baseline polygon (100% = baseline circle/hexagon)
                    val basePolygon = Path()
                    for (i in 0 until numAxes) {
                        val angle = Math.toRadians((i * 60 - 90).toDouble())
                        val r = radius * 0.7f // 0.7 radius represents 100% baseline norm
                        val x = centerX + (r * cos(angle)).toFloat()
                        val y = centerY + (r * sin(angle)).toFloat()
                        if (i == 0) basePolygon.moveTo(x, y) else basePolygon.lineTo(x, y)
                    }
                    basePolygon.close()
                    drawPath(
                        path = basePolygon,
                        color = outlineColor.copy(alpha = 0.4f),
                        style = Stroke(width = 1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 6f), 0f))
                    )

                    // Draw Current Week polygon
                    val currentPolygon = Path()
                    currentRatios.forEachIndexed { i, ratio ->
                        val angle = Math.toRadians((i * 60 - 90).toDouble())
                        val r = radius * 0.7f * ratio
                        val x = centerX + (r * cos(angle)).toFloat()
                        val y = centerY + (r * sin(angle)).toFloat()
                        if (i == 0) currentPolygon.moveTo(x, y) else currentPolygon.lineTo(x, y)
                    }
                    currentPolygon.close()

                    drawPath(
                        path = currentPolygon,
                        color = primaryColor.copy(alpha = 0.25f)
                    )
                    drawPath(
                        path = currentPolygon,
                        color = primaryColor,
                        style = Stroke(width = 2.dp.toPx())
                    )
                }
            }
        }
    }
}

@Composable
fun MiniSparkline(
    values: List<Float>,
    color: Color = MaterialTheme.colorScheme.primary,
    modifier: Modifier = Modifier.fillMaxWidth().height(40.dp)
) {
    if (values.isEmpty()) return
    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height
        val minV = values.minOrNull() ?: 0f
        val maxV = (values.maxOrNull() ?: 1f).coerceAtLeast(minV + 0.1f)
        val stepX = width / (values.size - 1).coerceAtLeast(1)

        val path = Path()
        values.forEachIndexed { i, v ->
            val x = i * stepX
            val norm = (v - minV) / (maxV - minV)
            val y = height * (1f - norm)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(
            path = path,
            color = color,
            style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
        )
    }
}
