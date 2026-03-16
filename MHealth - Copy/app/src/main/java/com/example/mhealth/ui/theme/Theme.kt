package com.example.mhealth.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = MintGreen,
    onPrimary = CardWhite,
    primaryContainer = SurfaceMint,
    onPrimaryContainer = TextPrimary,
    secondary = SkyBlue,
    onSecondary = TextPrimary,
    secondaryContainer = SurfaceBlue,
    onSecondaryContainer = TextPrimary,
    tertiary = LavenderPurple,
    onTertiary = TextPrimary,
    background = BackgroundWhite,
    onBackground = TextPrimary,
    surface = CardWhite,
    onSurface = TextPrimary,
    surfaceVariant = SurfaceBlue,
    onSurfaceVariant = TextSecondary,
    outline = TextMuted
)

@Composable
fun MHealthTheme(content: @Composable () -> Unit) {
    val colorScheme = LightColorScheme
    val view = LocalView.current

    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = BackgroundWhite.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = true
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}