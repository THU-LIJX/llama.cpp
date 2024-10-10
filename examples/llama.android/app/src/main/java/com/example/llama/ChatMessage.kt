package com.example.llama

import android.net.Uri

enum class role {
    USER, BOT
}

data class ChatMessage(
    val type: String, //"text", "image"
    val content: String,
    val imageUri: Uri? = null,
    val role: role,
)
