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
val Surface   = Color(0xFF1C1B1F)

private val DarkColors = darkColorScheme(
    primary        = Purple80,
    onPrimary      = Color(0xFF21005D),
    secondary      = Teal80,
    onSecondary    = Color(0xFF00201D),
    background     = Color(0xFF1C1B1F),
    surface        = Color(0xFF1C1B1F),
    surfaceVariant = Color(0xFF49454F),
    onBackground   = Color(0xFFE6E1E5),
    onSurface      = Color(0xFFE6E1E5),
)

private val LightColors = lightColorScheme(
    primary        = Purple40,
    onPrimary      = Color.White,
    secondary      = Teal40,
    onSecondary    = Color.White,
    background     = Color(0xFFFFFBFE),
    surface        = Color(0xFFFFFBFE),
    onBackground   = Color(0xFF1C1B1F),
    onSurface      = Color(0xFF1C1B1F),
)

@Composable
fun PocketLLMTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
