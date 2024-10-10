package com.example.llama

import android.content.ContentResolver
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class MainChatAppViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf<ChatMessage>())
        private set

    var generating by mutableStateOf(false)
        private set

    var message by mutableStateOf("")
        private set

    var imageUri: Uri? = null
        private set

    var contentResolver: ContentResolver? = null
        private set



    override fun onCleared() {
        super.onCleared()

        viewModelScope.launch {
            try {
//                llamaAndroid.unload()
//                llamaAndroid.unload_llava();
            } catch (exc: IllegalStateException) {
//                messages += exc.message!!
            }
        }
    }

    fun load_llava() {
        viewModelScope.launch {
            try {
                llamaAndroid.load_llava()
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load_llava() failed", exc)
            }
        }
    }

    fun load_image() {
        viewModelScope.launch {
            try {
                generating = true
                llamaAndroid.load_image(imageUri!!, contentResolver!!)
                Log.i(tag, "load_image() success")
                generating = false
            } catch (exc: IllegalStateException) {
                generating = false
                Log.e(tag, "load_image() failed", exc)
            }
        }
    }

    fun load() {
        viewModelScope.launch {
            try {
                llamaAndroid.load("/data/data/com.example.llama/files/ggml-model-q2_k.gguf")
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
            }
        }
    }

    fun generate() {
        val text = messages.last().content
        messages = messages + ChatMessage("text", "", role = role.BOT)
        viewModelScope.launch {
            generating = true
            llamaAndroid.send_llava(text)
                .catch {
                    Log.e(tag, "send() failed", it)
                }
                .collect {
                    var chat_message = ChatMessage("text", messages.last().content + it, role = role.BOT)
                    messages = messages.dropLast(1) + chat_message
                }
            generating = false
        }
    }

    fun addMessage(newMessage: ChatMessage) {
        messages = messages + newMessage
    }

    fun setImageUri(uri: Uri) {
        imageUri = uri
    }

    fun setContentResolver(contentResolver_: ContentResolver) {
        contentResolver = contentResolver_
    }

    fun updateGenerating(generating_: Boolean) {
        generating = generating_
    }

}
