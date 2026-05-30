package com.example.mhealth.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.darkColorScheme

private val LightColorScheme = lightColorScheme(
    primary = OceanBlue,
    onPrimary = CardWhite,
    primaryContainer = SurfaceMint,
    onPrimaryContainer = TextPrimary,
    secondary = SoftCyan,
    onSecondary = TextPrimary,
    secondaryContainer = SurfaceBlue,
    onSecondaryContainer = TextPrimary,
    tertiary = SereneTeal,
    onTertiary = TextPrimary,
    background = BackgroundWhite,
    onBackground = TextPrimary,
    surface = CardWhite,
    onSurface = TextPrimary,
    surfaceVariant = SurfaceBlue,
    onSurfaceVariant = TextSecondary,
    outline = TextMuted
)

private val DarkColorScheme = darkColorScheme(
    primary = LumenIndigo,
    onPrimary = LumenBgDark,
    primaryContainer = LumenSurfaceDark,
    onPrimaryContainer = LumenTextPrimaryDark,
    secondary = LumenLavender,
    onSecondary = LumenBgDark,
    secondaryContainer = LumenCardDark,
    onSecondaryContainer = LumenTextPrimaryDark,
    tertiary = LumenTeal,
    onTertiary = LumenBgDark,
    background = LumenBgDark,
    onBackground = LumenTextPrimaryDark,
    surface = LumenSurfaceDark,
    onSurface = LumenTextPrimaryDark,
    surfaceVariant = LumenCardDark,
    onSurfaceVariant = LumenTextSecondaryDark,
    outline = LumenTextMutedDark
)

@Composable
fun CoveTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme
    val view = LocalView.current

    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = (if (darkTheme) LumenBgDark else BackgroundWhite).toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
