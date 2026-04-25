package com.nomad.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColors = darkColorScheme(
    primary        = Color.White,
    onPrimary      = Color.Black,
    secondary      = Color.White,
    onSecondary    = Color.Black,
    background     = Color.Black,
    surface        = Color.Black,
    surfaceVariant = Color.Black,
    onBackground   = Color.White,
    onSurface      = Color.White,
    outline        = Color.White
)

private val LightColors = darkColorScheme(
    primary        = Color.White,
    onPrimary      = Color.Black,
    secondary      = Color.White,
    onSecondary    = Color.Black,
    background     = Color.Black,
    surface        = Color.Black,
    surfaceVariant = Color.Black,
    onBackground   = Color.White,
    onSurface      = Color.White,
    outline        = Color.White
)

@Composable
fun NomadTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
