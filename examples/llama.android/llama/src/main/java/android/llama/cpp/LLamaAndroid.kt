package android.llama.cpp

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    private val tag: String? = this::class.simpleName

    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("llama-android")

            // Set llama log handler to Android
            log_to_android()
            backend_init(false)

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private val nlen: Int = 128

    private external fun log_to_android()
    private external fun load_model(filename: String): Long
    private external fun free_model(model: Long)
    private external fun new_context(model: Long): Long
    private external fun free_context(context: Long)
    private external fun backend_init(numa: Boolean)
    private external fun backend_free()
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batch: Long)
    private external fun new_sampler(): Long
    private external fun free_sampler(sampler: Long)
    private external fun bench_model(
        context: Long,
        model: Long,
        batch: Long,
        pp: Int,
        tg: Int,
        pl: Int,
        nr: Int
    ): String

    private external fun system_info(): String

    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        nLen: Int
    ): Int

    private external fun completion_loop(
        context: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun kv_cache_clear(context: Long)

    // functions in llava-cli.cpp

    private external fun load_image(ctx_llava_pointer: Long): Long
    private external fun load_image_with_btyes(ctx_llava_pointer: Long, byteArray: ByteArray): Long
    private external fun llava_init_context(): Long
    private external fun llava_init_sampler(): Long
    private external fun llava_completion_init(ctx_llava_pointer: Long, image_embed_pointer: Long, system_prompt: String, user_prompt: String): Int
    private external fun llava_completion_loop(sampler_pointer: Long, image_embed_pointer: Long, ctx_llava_pointer: Long, n_past_tokens: Int): String?

    private external fun llava_get_default_gpt_params(): Long
    private external fun llava_init(params_gpt: Long): Long
    private external fun llava_free(ctx_llava: Long)
    private external fun llama_free_model(model: Long)
    private external fun llava_image_embed_free(image_embed: Long)
    private external fun llava_free_sampler(sampler: Long)

    private external fun llava_debug()
    private external fun llava_debug2()


    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    Log.d(tag, "bench(): $state")
                    bench_model(state.context, state.model, state.batch, pp, tg, pl, nr)
                }

                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun load(pathToModel: String) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    val model = load_model(pathToModel)
                    if (model == 0L)  throw IllegalStateException("load_model() failed")

                    val context = new_context(model)
                    if (context == 0L) throw IllegalStateException("new_context() failed")

                    val batch = new_batch(512, 0, 1)
                    if (batch == 0L) throw IllegalStateException("new_batch() failed")

                    val sampler = new_sampler()
                    if (sampler == 0L) throw IllegalStateException("new_sampler() failed")

                    Log.i(tag, "Loaded model $pathToModel")
                    threadLocalState.set(State.Loaded(model, context, batch, sampler))

//                    completion_init(context, batch, "Hello", nlen)
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    suspend fun load_llava() {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {

                    val ctx_llava = llava_init_context()
                    if (ctx_llava == 0L) throw IllegalStateException("llava_init_context() failed")

//                    val image_embed = load_image(ctx_llava)
//                    if (image_embed == 0L) throw IllegalStateException("load_image() failed")

                    val sampler = llava_init_sampler()
                    if (sampler == 0L) throw IllegalStateException("sampler() failed")

                    Log.i(tag, "Loaded model llava")
                    threadLocalState.set(State.LoadedLlava(ctx_llava, 0, sampler))

//                    Log.i(tag, "llava_completion_init() start")
//                    val system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:"
//                    val user_prompt = "What's the main content of this photo?" + "\nASSISTANT:"
//                    val n = llava_completion_init(ctx_llava, image_embed, system_prompt, user_prompt)
//                    Log.i(tag, "n_past $n")
//
//                    llava_completion_loop(sampler, ctx_llava, n)

                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    suspend fun load_image(imageUri: Uri, contentResolver: ContentResolver) {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.LoadedLlava -> {
//                    val ctx_llava = llava_init_context()
//                    if (ctx_llava == 0L) throw IllegalStateException("llava_init_context() failed")

//                    val sampler = llava_init_sampler()
//                    if (sampler == 0L) throw IllegalStateException("sampler() failed")

                    val inputStream: InputStream? = contentResolver.openInputStream(imageUri);
                    val outputStream: ByteArrayOutputStream = ByteArrayOutputStream();
                    val bufferSize = 1024;
                    val buffer = ByteArray(bufferSize)

                    var len = 0
                    while (inputStream?.read(buffer).also { len = it ?: -1 } != -1) {
                        outputStream.write(buffer, 0, len)
                    }
                    println("len: ${outputStream.toByteArray().size}")

                    val image_embed = load_image_with_btyes(state.ctx_llava, outputStream.toByteArray())
//                    val image_embed = load_image_with_btyes(ctx_llava, outputStream.toByteArray())

                    Log.i(tag, "Loaded model llava")
                    threadLocalState.set(State.LoadedLlava(state.ctx_llava, image_embed, state.sampler))
//                    threadLocalState.set(State.LoadedLlava(ctx_llava, image_embed, sampler))

                }
                else -> throw IllegalStateException("Model has not been loaded")
            }
        }
    }

    suspend fun unload_llava() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.LoadedLlava -> {
                    llava_image_embed_free(state.image_embed)
                    llava_free(state.ctx_llava)
                }
                else -> {}
            }
        }
    }

    fun send(message: String): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                val ncur = IntVar(completion_init(state.context, state.batch, message, nlen))
                while (ncur.value <= nlen) {
                    val str = completion_loop(state.context, state.batch, state.sampler, nlen, ncur)
                    if (str == null) {
                        break
                    }
                    emit(str)
                }
                kv_cache_clear(state.context)
            }
            else -> {}
        }
    }.flowOn(runLoop)

    fun send_llava(message: String): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.LoadedLlava -> {
                val system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:"
                val user_prompt = message + "\nASSISTANT:"
//                val user_prompt = "What's the main content of this photo?" + "\nASSISTANT:"
//                val user_prompt = "describe the image in detail" + "\nASSISTANT:"
                val n = llava_completion_init(state.ctx_llava, state.image_embed, system_prompt, user_prompt)
                Log.i(tag, "n_past $n")
                var i = 0
                while (i < 64) {
                    val str = llava_completion_loop(state.sampler, state.image_embed, state.ctx_llava, n+i);
                    if (str == null) {
                        break
                    }
                    i += 1
                    emit(str)
                }
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    free_context(state.context)
                    free_model(state.model)
                    free_batch(state.batch)
                    free_sampler(state.sampler);

                    threadLocalState.set(State.Idle)
                }
                else -> {}
            }
        }
    }

    companion object {
        private class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set

            fun inc() {
                synchronized(this) {
                    value += 1
                }
            }
        }

        private sealed interface State {
            data object Idle: State
            data class Loaded(val model: Long, val context: Long, val batch: Long, val sampler: Long): State
            data class LoadedLlava(val ctx_llava: Long, val image_embed: Long, val sampler: Long): State
        }

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = _instance
    }
}
