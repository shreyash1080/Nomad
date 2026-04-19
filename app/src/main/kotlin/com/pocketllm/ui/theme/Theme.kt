package com.pocketllm.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Brand colours
val Purple80  = Color(0xFFD0BCFF)
val Purple40  = Color(0xFF6650A4)
val Teal80    = Color(0xFF80CBC4)
val Teal40    = Color(0xFF00897B)

private val DarkColors = darkColorScheme(
    primary        = Color.White,
    onPrimary      = Color.Black,
    secondary      = Color(0xFF333333),
    onSecondary    = Color.White,
    background     = Color.Black,
    surface        = Color.Black,
    surfaceVariant = Color(0xFF111111),
    onBackground   = Color.White,
    onSurface      = Color.White,
    outline        = Color(0xFF333333)
)

private val LightColors = darkColorScheme( // Force dark even in "light" mode for the "Eigen" aesthetic
    primary        = Color.White,
    onPrimary      = Color.Black,
    secondary      = Color(0xFF333333),
    onSecondary    = Color.White,
    background     = Color.Black,
    surface        = Color.Black,
    surfaceVariant = Color(0xFF111111),
    onBackground   = Color.White,
    onSurface      = Color.White,
    outline        = Color(0xFF333333)
)

@Composable
fun EigenTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
