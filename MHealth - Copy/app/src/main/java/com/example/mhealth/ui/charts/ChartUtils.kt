package com.example.mhealth.ui.charts

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import android.content.pm.PackageManager
import com.example.mhealth.ui.theme.*
import kotlin.math.*

// =============================================================================
// ArcProgressRing — animated arc gauge for a single metric
// =============================================================================
@Composable
fun ArcProgressRing(
    value: Float,
    maxValue: Float,
    color: Color,
    label: String,
    unit: String = "",
    size: Dp = 100.dp,
    strokeWidth: Dp = 10.dp
) {
    val animProgress = remember { Animatable(0f) }
    LaunchedEffect(value) {
        animProgress.animateTo(
            (value / maxValue.coerceAtLeast(0.01f)).coerceIn(0f, 1f),
            animationSpec = tween(1000, easing = FastOutSlowInEasing)
        )
    }
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(contentAlignment = Alignment.Center, modifier = Modifier.size(size)) {
            Canvas(modifier = Modifier.size(size)) {
                val stroke = strokeWidth.toPx()
                val inset = stroke / 2
                val arcSize = Size(this.size.width - stroke, this.size.height - stroke)
                val startAngle = 135f
                val sweepTotal = 270f
                // Track
                drawArc(
                    color = color.copy(alpha = 0.15f),
                    startAngle = startAngle,
                    sweepAngle = sweepTotal,
                    useCenter = false,
                    topLeft = Offset(inset, inset),
                    size = arcSize,
                    style = Stroke(stroke, cap = StrokeCap.Round)
                )
                // Fill
                drawArc(
                    brush = Brush.sweepGradient(
                        listOf(color.copy(alpha = 0.7f), color),
                        center = Offset(this.size.width / 2, this.size.height / 2)
                    ),
                    startAngle = startAngle,
                    sweepAngle = sweepTotal * animProgress.value,
                    useCenter = false,
                    topLeft = Offset(inset, inset),
                    size = arcSize,
                    style = Stroke(stroke, cap = StrokeCap.Round)
                )
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = if (maxValue <= 1f) "%.0f%%".format(value * 100) else "%.1f".format(value),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = color
                )
                if (unit.isNotBlank()) Text(unit, fontSize = 10.sp, color = TextSecondary)
            }
        }
        Spacer(Modifier.height(4.dp))
        Text(label, fontSize = 11.sp, color = TextSecondary, textAlign = TextAlign.Center)
    }
}

// =============================================================================
// SparklineChart — mini multi-point line chart with gradient fill
// =============================================================================
@Composable
fun SparklineChart(
    values: List<Float>,
    color: Color,
    modifier: Modifier = Modifier,
    showDots: Boolean = true
) {
    if (values.size < 2) {
        Canvas(modifier = modifier) {
            drawLine(color.copy(alpha = 0.3f), Offset(0f, size.height / 2), Offset(size.width, size.height / 2), 2f)
        }
        return
    }
    Canvas(modifier = modifier) {
        val min = values.min()
        val max = values.max()
        val range = (max - min).coerceAtLeast(0.01f)
        val xStep = size.width / (values.size - 1)

        fun xAt(i: Int) = i * xStep
        fun yAt(v: Float) = size.height - ((v - min) / range) * size.height * 0.85f - size.height * 0.075f

        // Gradient fill path
        val fillPath = Path().apply {
            moveTo(xAt(0), size.height)
            values.forEachIndexed { i, v -> lineTo(xAt(i), yAt(v)) }
            lineTo(xAt(values.lastIndex), size.height)
            close()
        }
        drawPath(fillPath, Brush.verticalGradient(listOf(color.copy(0.3f), color.copy(0.0f))))

        // Line path
        val linePath = Path().apply {
            values.forEachIndexed { i, v ->
                if (i == 0) moveTo(xAt(i), yAt(v)) else lineTo(xAt(i), yAt(v))
            }
        }
        drawPath(linePath, color, style = Stroke(3f, cap = StrokeCap.Round))

        // Dots
        if (showDots) {
            values.forEachIndexed { i, v ->
                drawCircle(color, 4f, Offset(xAt(i), yAt(v)))
                drawCircle(Color.White, 2f, Offset(xAt(i), yAt(v)))
            }
        }
    }
}

// =============================================================================
// RadarChart — hexagonal radar with 6 features
// =============================================================================
@Composable
fun RadarChart(
    labels: List<String>,
    values: List<Float>,                    // normalised 0-1 (current vs baseline)
    baseline: List<Float>,                  // normalised 0-1 (personal baseline)
    color: Color = MintGreen,
    modifier: Modifier = Modifier,
    prototypeValues: List<Float>? = null    // 0-1 prototype shape — drawn ONLY when non-null
) {
    val animProgress = remember { Animatable(0f) }
    LaunchedEffect(values) {
        animProgress.animateTo(1f, tween(1200, easing = FastOutSlowInEasing))
    }

    Canvas(modifier = modifier) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val maxRadius = minOf(cx, cy) * 0.90f
        val n = labels.size
        val angleStep = 2 * PI / n

        fun polarOffset(i: Int, r: Float): Offset {
            val angle = -PI / 2 + i * angleStep
            return Offset(cx + r * cos(angle).toFloat(), cy + r * sin(angle).toFloat())
        }

        // Grid rings (3 levels)
        for (level in 1..3) {
            val r = maxRadius * level / 3
            val path = Path()
            for (i in 0 until n) {
                val pt = polarOffset(i, r)
                if (i == 0) path.moveTo(pt.x, pt.y) else path.lineTo(pt.x, pt.y)
            }
            path.close()
            drawPath(path, Color.Gray.copy(0.15f), style = Stroke(1f))
        }

        // Spokes
        for (i in 0 until n) {
            drawLine(Color.Gray.copy(0.2f), Offset(cx, cy), polarOffset(i, maxRadius), 1f)
        }

        // ── Baseline polygon (sky-blue fill) ────────────────────────────────
        val baselinePath = Path()
        for (i in 0 until n) {
            val r = maxRadius * (baseline.getOrElse(i) { 0.5f }).coerceIn(0f, 1f)
            val pt = polarOffset(i, r)
            if (i == 0) baselinePath.moveTo(pt.x, pt.y) else baselinePath.lineTo(pt.x, pt.y)
        }
        baselinePath.close()
        drawPath(baselinePath, SkyBlue.copy(0.15f))
        drawPath(baselinePath, SkyBlue.copy(0.5f), style = Stroke(2f))

        // ── Prototype polygon (dashed red) — only when a disorder is matched ─
        if (prototypeValues != null) {
            val dashEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f), 0f)
            val prototypePath = Path()
            for (i in 0 until n) {
                val r = maxRadius * (prototypeValues.getOrElse(i) { 0.5f }).coerceIn(0f, 1f)
                val pt = polarOffset(i, r)
                if (i == 0) prototypePath.moveTo(pt.x, pt.y) else prototypePath.lineTo(pt.x, pt.y)
            }
            prototypePath.close()
            // Subtle fill so the shape is visible without overwhelming the chart
            drawPath(prototypePath, Color(0xFFEF5350).copy(alpha = 0.08f))
            drawPath(
                prototypePath,
                Color(0xFFEF5350).copy(alpha = 0.85f),
                style = Stroke(width = 2.2f, pathEffect = dashEffect)
            )
        }

        // ── Current polygon (animated, colour-filled) ────────────────────────
        val valuePath = Path()
        for (i in 0 until n) {
            val r = maxRadius * (values.getOrElse(i) { 0f }).coerceIn(0f, 1f) * animProgress.value
            val pt = polarOffset(i, r)
            if (i == 0) valuePath.moveTo(pt.x, pt.y) else valuePath.lineTo(pt.x, pt.y)
        }
        valuePath.close()
        drawPath(valuePath, color.copy(0.25f))
        drawPath(valuePath, color.copy(0.9f), style = Stroke(2.5f, cap = StrokeCap.Round))

        // Vertex dots
        for (i in 0 until n) {
            val r = maxRadius * (values.getOrElse(i) { 0f }).coerceIn(0f, 1f) * animProgress.value
            val pt = polarOffset(i, r)
            drawCircle(color, 5f, pt)
            drawCircle(Color.White, 2.5f, pt)
        }
    }
}

// =============================================================================
// HorizontalBarChart — ranked horizontal bars
// =============================================================================
@Composable
fun HorizontalBarChart(
    items: List<Pair<String, Float>>,  // label → value
    maxValue: Float,
    color: Color = MintGreen,
    unitSuffix: String = "",
    modifier: Modifier = Modifier
) {
    val animProgress = remember { Animatable(0f) }
    LaunchedEffect(items) { animProgress.animateTo(1f, tween(900)) }

    val pm = LocalContext.current.packageManager

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        items.take(6).forEach { (label, value) ->
            val fraction = (value / maxValue.coerceAtLeast(0.01f)) * animProgress.value
            
            val appName = try {
                val appInfo = pm.getApplicationInfo(label, 0)
                pm.getApplicationLabel(appInfo).toString()
            } catch (e: PackageManager.NameNotFoundException) {
                label.substringAfterLast('.').replaceFirstChar { it.uppercase() }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    appName.take(14),
                    fontSize = 11.sp, color = TextSecondary,
                    modifier = Modifier.width(100.dp)
                )
                Box(Modifier.weight(1f).height(14.dp)) {
                    Canvas(Modifier.fillMaxSize()) {
                        drawRoundRect(color.copy(0.15f), cornerRadius = androidx.compose.ui.geometry.CornerRadius(7f))
                        drawRoundRect(
                            Brush.horizontalGradient(listOf(color.copy(0.8f), color)),
                            size = androidx.compose.ui.geometry.Size(size.width * fraction, size.height),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(7f)
                        )
                    }
                }
                Spacer(Modifier.width(6.dp))
                Text("${value.toInt()}$unitSuffix", fontSize = 10.sp, color = color, fontWeight = FontWeight.SemiBold, modifier = Modifier.width(36.dp))
            }
        }
    }
}

// =============================================================================
// AnomalyScoreGauge — half-circle gauge with color zones
// =============================================================================
@Composable
fun AnomalyScoreGauge(
    score: Float,           // 0-1
    modifier: Modifier = Modifier
) {
    val animScore = remember { Animatable(0f) }
    LaunchedEffect(score) { animScore.animateTo(score.coerceIn(0f, 1f), tween(1200)) }

    Canvas(modifier = modifier) {
        val cx = size.width / 2; val cy = size.height * 0.85f
        val radius = minOf(cx, cy) * 0.85f
        val stroke = radius * 0.22f

        // Colored zone arcs (left to right: green → yellow → orange → red)
        val zones = listOf(
            AlertGreen to 0.25f, AlertYellow to 0.25f, AlertOrange to 0.25f, AlertRed to 0.25f
        )
        var startA = 180f
        zones.forEach { (c, fraction) ->
            val sweep = 180f * fraction
            drawArc(c, startA, sweep, false,
                Offset(cx - radius, cy - radius), Size(2 * radius, 2 * radius),
                style = Stroke(stroke, cap = StrokeCap.Butt))
            startA += sweep
        }

        // Needle
        val needleAngle = Math.toRadians((180 + animScore.value * 180).toDouble())
        val needleLen = radius * 0.7f
        drawLine(
            TextPrimary, Offset(cx, cy),
            Offset(cx + needleLen * cos(needleAngle).toFloat(), cy + needleLen * sin(needleAngle).toFloat()),
            strokeWidth = stroke * 0.2f, cap = StrokeCap.Round
        )
        drawCircle(TextPrimary, stroke * 0.35f, Offset(cx, cy))
        drawCircle(Color.White, stroke * 0.18f, Offset(cx, cy))
    }
}

// =============================================================================
// PieChart — animated donut/pie chart for categorical distribution
// =============================================================================
@Composable
fun PieChart(
    data: Map<String, Float>,
    colors: Map<String, Color>,
    icons: Map<String, String>? = null,
    centerText: String = "",
    centerSubtext: String = "",
    modifier: Modifier = Modifier
) {
    val total = data.values.sum().coerceAtLeast(0.01f)
    val animProgress = remember { Animatable(0f) }
    
    LaunchedEffect(data) {
        animProgress.animateTo(
            1f, 
            animationSpec = tween(1200, easing = FastOutSlowInEasing)
        )
    }

    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Pie Chart Canvas (Donut Style)
        Box(
            modifier = Modifier.size(150.dp),
            contentAlignment = Alignment.Center
        ) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val strokeWidth = 28.dp.toPx()
                val radius = size.minDimension / 2f - strokeWidth / 2f
                var startAngle = -90f

                data.forEach { (key, value) ->
                    val sweepAngle = (value / total) * 360f
                    val sweepAnimated = sweepAngle * animProgress.value
                    
                    if (sweepAnimated > 0.5f) {
                        drawArc(
                            color = colors[key] ?: Color.Gray,
                            startAngle = startAngle,
                            sweepAngle = sweepAnimated,
                            useCenter = false,
                            topLeft = Offset(size.width / 2 - radius, size.height / 2 - radius),
                            size = Size(radius * 2, radius * 2),
                            style = Stroke(strokeWidth, cap = StrokeCap.Butt)
                        )
                        startAngle += sweepAnimated + 1.5f // 1.5度 gap for sleek separation
                    }
                }
            }
            if (centerText.isNotEmpty()) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(centerText, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, color = TextPrimary)
                    if (centerSubtext.isNotEmpty()) {
                        Text(centerSubtext, fontSize = 10.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.width(20.dp))

        // Legend beside the pie
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            data.entries.sortedByDescending { it.value }.take(5).forEach { (key, value) ->
                val icon = icons?.get(key) ?: "▪️"
                val color = colors[key] ?: Color.Gray
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Canvas(Modifier.size(8.dp)) { drawCircle(color) }
                    Spacer(Modifier.width(8.dp))
                    Text("$icon $key", fontSize = 11.sp, color = color, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Text(
                        if (value % 1 == 0f) "${value.toInt()}" else "%.1f".format(value), 
                        fontSize = 12.sp, color = TextPrimary, fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}
