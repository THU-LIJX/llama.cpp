package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentResolver
import android.graphics.BitmapFactory
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.net.URI

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

//    private val viewModel: MainViewModel by viewModels()
    private val viewModel: MainChatAppViewModel by viewModels()

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        viewModel.setContentResolver(contentResolver)
//        viewModel.load()
        viewModel.load_llava()

//        val free = Formatter.formatFileSize(this, availableMemory().availMem)
//        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
//
//        viewModel.log("Current memory: $free / $total")
//        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        requestPermissions(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE), 0)
        requestPermissions(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO), 0)

        val extFilesDir = getExternalFilesDir(null)

//        val models = listOf(
//            Downloadable(
//                "Phi-2 7B (Q4_0, 1.6 GiB)",
//                Uri.parse("https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true"),
//                File(extFilesDir, "phi-2-q4_0.gguf"),
//            ),
//            Downloadable(
//                "TinyLlama 1.1B (f16, 2.2 GiB)",
//                Uri.parse("https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true"),
//                File(extFilesDir, "tinyllama-1.1-f16.gguf"),
//            ),
//            Downloadable(
//                "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
//                Uri.parse("https://huggingface.co/TheBloke/phi-2-dpo-GGUF/resolve/main/phi-2-dpo.Q3_K_M.gguf?download=true"),
//                File(extFilesDir, "phi-2-dpo.Q3_K_M.gguf")
//            ),
//        )

        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
//                    MainCompose(
//                        viewModel,
//                        clipboardManager,
//                        downloadManager,
//                        models,
//                    )
                    ChatApp(viewModel, contentResolver)
                }

            }
        }
    }
}

@Composable
fun ChatApp(
    viewModel: MainChatAppViewModel,
    contentResolver: ContentResolver
) {

//    var messages by remember {
//        mutableStateOf(listOf<ChatMessage>())
//    }
//    var generating by remember {
//        mutableStateOf(false);
//    }


    val pickImageLauncher = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            viewModel.addMessage(ChatMessage("image", "", it, role.USER))
            viewModel.setImageUri(uri)
            viewModel.load_image()
//            val inputStream: InputStream? = contentResolver.openInputStream(uri);
//            val outputStream: ByteArrayOutputStream = ByteArrayOutputStream();
//            val bufferSize = 1024;
//            val buffer = ByteArray(bufferSize)
//
//            var len = 0
//            while (inputStream?.read(buffer).also { len = it ?: -1 } != -1) {
//                outputStream.write(buffer, 0, len)
//            }
//            println("len: ${outputStream.toByteArray().size}")
        }
    }

    Column {
        if (viewModel.generating) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.3f)
            )
        }
        Chat(
            messages = viewModel.messages,
            selectImage = {
                pickImageLauncher.launch("image/*")
            },
            generating = viewModel.generating
        ) {
            viewModel.addMessage(ChatMessage("text", it, role = role.USER))
            viewModel.generate()
        }
    }

}

fun simulateStreamingOutput(coroutineScope: CoroutineScope, onOutput: (String) -> Unit, onFinish: () -> Unit) {
    // Simulate streaming output by sending incremental content over time
    val content = "This is a simulated streaming output from the bot.This is a simulated streaming output from the bot."
    val chunks = content.chunked(10) // Split into chunks

    coroutineScope.launch {
        for (i in chunks.indices) {
            kotlinx.coroutines.delay(1000)
            onOutput(chunks.take(i + 1).joinToString(""))
        }
        onFinish()
    }
}


@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    dm: DownloadManager,
    models: List<Downloadable>
) {


//    val pickImageLauncher = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent()) { uri: Uri? ->
//
//        uri?.let {
//
//            if(messages.isNotEmpty()) {
//                //clear messages
//                messages = emptyList()
//                messages = messages + ChatMessage("image", "", it, role.USER)
//
//            }
//            else {
//                messages = messages + ChatMessage("image", "", it, role.USER)
//            }
//        }
//    }

    Column {
        val scrollState = rememberLazyListState()

        Box(modifier = Modifier.weight(1f)) {
            LazyColumn(state = scrollState) {
                items(viewModel.messages) {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodyLarge.copy(color = LocalContentColor.current),
                        modifier = Modifier.padding(16.dp)
                    )
                }
            }
        }
        OutlinedTextField(
            value = viewModel.message,
            onValueChange = { viewModel.updateMessage(it) },
            label = { Text("Message") },
        )
        Row {
//            Button({ viewModel.send() }) { Text("Send") }
            Button({ viewModel.send_llava() }) { Text("Send") }

//            Button({ pickImageLauncher.launch("image/*") }) { Text("Select") }
//            Button({ viewModel.bench(8, 4, 1) }) { Text("Bench") }
//            Button({ viewModel.clear() }) { Text("Clear") }
//            Button({
//                viewModel.messages.joinToString("\n").let {
//                    clipboard.setPrimaryClip(ClipData.newPlainText("", it))
//                }
//            }) { Text("Copy") }
//            Button({ viewModel.load("/data/data/com.example.llama/files/qwen2-0_5b-instruct-q8_0.gguf") }) { Text("Load") }
//            Button({ viewModel.load("/data/data/com.example.llama/files/ggml-model-q2_k.gguf") }) { Text("Load") }
//            Button({ viewModel.load("/data/data/com.example.llama/files/ggml-model-q4_k.gguf") }) { Text("Load") }
//            Button({ viewModel.load("/data/data/com.example.llama/files/phi-2.8B-2-Q4_K_M.gguf") }) { Text("Load") }

            Button({ viewModel.load_llava() }) { Text("Load Llava") }
        }

//        Column {
//            for (model in models) {
//                Downloadable.Button(viewModel, dm, model)
//            }
//        }
    }
}
