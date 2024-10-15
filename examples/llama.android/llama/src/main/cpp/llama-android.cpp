#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include <sampling.h>
#include "llama.h"
#include "common.h"
#include "llava-cli.h"

// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("llama-android");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("llama-android")
//      }
//    }

#define TAG "llama-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    if (level == GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1model(JNIEnv *env, jobject, jstring filename) {
    llama_model_params model_params = llama_model_default_params();

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s", path_to_model);

    auto model = llama_load_model_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1model(JNIEnv *, jobject, jlong model) {
    llama_free_model(reinterpret_cast<llama_model *>(model));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1context(JNIEnv *env, jobject, jlong jmodel) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 2048;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context * context = llama_new_context_with_model(model, ctx_params);

    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
        return 0;
    }

    return reinterpret_cast<jlong>(context);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1context(JNIEnv *, jobject, jlong context) {
    llama_free(reinterpret_cast<llama_context *>(context));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_bench_1model(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong batch_pointer,
        jint pp,
        jint tg,
        jint pl,
        jint nr
        ) {
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto model = reinterpret_cast<llama_model *>(model_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    const int n_ctx = llama_n_ctx(context);

    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp)");

        llama_batch_clear(*batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            llama_batch_add(*batch, 0, i, { 0 }, false);
        }

        batch->logits[batch->n_tokens - 1] = true;
        llama_kv_cache_clear(context);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, *batch) != 0) {
            LOGi("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg)");

        llama_kv_cache_clear(context);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {

            llama_batch_clear(*batch);
            for (j = 0; j < pl; j++) {
                llama_batch_add(*batch, 0, i, { j }, true);
            }

            LOGi("llama_decode() text generation: %d", i);
            if (llama_decode(context, *batch) != 0) {
                LOGi("llama_decode() failed during text generation");
            }
        }

        const auto t_tg_end = ggml_time_us();

        llama_kv_cache_clear(context);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(model)) / 1e9;

    const auto backend    = "(Android)"; // TODO: What should this be?

    std::stringstream result;
    result << std::setprecision(2);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";

    return env->NewStringUTF(result.str().c_str());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1batch(JNIEnv *, jobject, jint n_tokens, jint embd, jint n_seq_max) {

    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.

    llama_batch *batch = new llama_batch {
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        0,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    return reinterpret_cast<jlong>(batch);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1batch(JNIEnv *, jobject, jlong batch_pointer) {
    llama_batch_free(*reinterpret_cast<llama_batch *>(batch_pointer));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1init(JNIEnv *, jobject) {
    llama_backend_init();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1init(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jint n_len
    ) {

    cached_token_chars.clear();

    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    const auto tokens_list = llama_tokenize(context, text, 1);

    auto n_ctx = llama_n_ctx(context);
    auto n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOGi("n_len = %d, n_ctx = %d, n_kv_req = %d", n_len, n_ctx, n_kv_req);

    if (n_kv_req > n_ctx) {
        LOGe("error: n_kv_req > n_ctx, the required KV cache size is not big enough");
    }

    for (auto id : tokens_list) {
        LOGi("%s", llama_token_to_piece(context, id).c_str());
    }

    llama_batch_clear(*batch);

    // evaluate the initial prompt
    for (auto i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(*batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() failed");
    }

    env->ReleaseStringUTFChars(jtext, text);

    return batch->n_tokens;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1loop(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
) {
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch   = reinterpret_cast<llama_batch   *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);
    const auto model = llama_get_model(context);

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    // sample the most likely token
    const auto new_token_id = llama_sampler_sample(sampler, context, -1);

    const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);
    if (llama_token_is_eog(model, new_token_id) || n_cur == n_len) {
        return nullptr;
    }

    auto new_token_chars = llama_token_to_piece(context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring new_token = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        new_token = env->NewStringUTF(cached_token_chars.c_str());
        LOGi("cached: %s, new_token_chars: `%s`, id: %d", cached_token_chars.c_str(), new_token_chars.c_str(), new_token_id);
        cached_token_chars.clear();
    } else {
        new_token = env->NewStringUTF("");
    }

    llama_batch_clear(*batch);
    llama_batch_add(*batch, new_token_id, n_cur, { 0 }, true);

    env->CallVoidMethod(intvar_ncur, la_int_var_inc);

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() returned null");
    }

    return new_token;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_kv_1cache_1clear(JNIEnv *, jobject, jlong context) {
    llama_kv_cache_clear(reinterpret_cast<llama_context *>(context));
}

//extern "C"
//JNIEXPORT void JNICALL
//Java_android_llama_cpp_LLamaAndroid_free_1model(JNIEnv *, jobject, jlong model) {
//    llama_free_model(reinterpret_cast<llama_model *>(model));
//}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1image(JNIEnv *, jobject, jlong ctx_llava_pointer) {
    const char * path_to_image = "/data/data/com.example.llama/files/cat.jpeg";
    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    auto ctx_llava = reinterpret_cast<llava_context *>(ctx_llava_pointer);


    llava_image_embed * image_embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, n_threads, path_to_image);
    if (!image_embed) {
        LOGe("image embed failed");
    } else {
        LOGi("image embed success");
    }
    return reinterpret_cast<jlong>(image_embed);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1image_1with_1btyes(JNIEnv * env, jobject, jlong ctx_llava_pointer, jbyteArray byte_array) {
    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    auto ctx_llava = reinterpret_cast<llava_context *>(ctx_llava_pointer);

    jsize length = env->GetArrayLength(byte_array);
    jbyte  * native_byte_array = env->GetByteArrayElements(byte_array, NULL);

    int len = (int) length;
    unsigned  char * image_bytes = (unsigned char *) native_byte_array;
    llava_image_embed * image_embed = llava_image_embed_make_with_bytes(ctx_llava->ctx_clip, n_threads, image_bytes, len);
    if (!image_embed) {
        LOGe("image embed failed");
    } else {
        LOGi("image embed success");
    }
    return reinterpret_cast<jlong>(image_embed);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1get_1default_1gpt_1params(JNIEnv *, jobject) {
    gpt_params * params = new gpt_params;
    params->n_ubatch = 1024;
    params->model = "/data/data/com.example.llama/files/ggml-model-q4_k.gguf";
//    params->model = "/data/data/com.example.llama/files/ggml-model-q2_k.gguf";
//    params->model = "/data/data/com.example.llama/files/qwen2-0_5b-instruct-q8_0.gguf";
    params->mmproj = "/data/data/com.example.llama/files/mmproj-model-f16.gguf";
    params->n_predict = 128;
    params->image.emplace_back("/data/data/com.example.llama/files/cat.jpeg");
    gpt_init();
    return reinterpret_cast<jlong>(params);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1init(JNIEnv * env, jobject, jlong params_gpt) {
//    auto llama_model = llava_init(reinterpret_cast<struct gpt_params *>(params_gpt));
//    return reinterpret_cast<jlong>(llama_model);
    gpt_params * params = reinterpret_cast<struct gpt_params *>(params_gpt);
    llama_model_params model_params = llama_model_params_from_gpt_params(*params);
    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1init_1context(JNIEnv * env, jobject) {
    llama_model_params model_params = llama_model_default_params();
//    const char * path_to_model = "/data/data/com.example.llama/files/qwen2-0_5b-instruct-q8_0.gguf";
//    const char * path_to_model = "/data/data/com.example.llama/files/ggml-model-q2_k.gguf";
    const char * path_to_model = "/data/data/com.example.llama/files/ggml-model-q4_k.gguf";
    const char * clip_path = "/data/data/com.example.llama/files/mmproj-model-f16.gguf";

    auto model = llama_load_model_from_file(path_to_model, model_params);
    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
    } else {
        LOGi("load_model() success");
    }

    auto ctx_clip = clip_model_load(clip_path, 1);
    if (!ctx_clip) {
        LOGe("clip_model_load() failed");
    } else {
        LOGi("clip_model_load() success");
    }

    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 2048;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_ubatch        = 1024;

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);
    if (!ctx_llama) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
    } else {
        LOGi("llama_new_context_with_model() finished");
    }

    auto * ctx_llava = (struct llava_context *) malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;

    return reinterpret_cast<jlong>(ctx_llava);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1init_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1free(JNIEnv *, jobject, jlong ctx_llava) {
    llava_free(reinterpret_cast<struct llava_context *>(ctx_llava));
}
extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llama_1free_1model(JNIEnv *, jobject, jlong model) {
    llama_free_model(reinterpret_cast<struct llama_model*>(model));
}
extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1image_1embed_1free(JNIEnv *, jobject, jlong image_embed) {
    llava_image_embed_free(reinterpret_cast<struct llava_image_embed*>(image_embed));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1free_1sampler(JNIEnv *, jobject, jlong sampler) {
    gpt_sampler_free(reinterpret_cast<gpt_sampler *>(sampler));
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1completion_1init(JNIEnv * env, jobject, jlong ctx_llava_pointer, jlong image_embed_pointer, jstring system_prompt, jstring user_prompt) {
    auto ctx_llava = reinterpret_cast<llava_context *>(ctx_llava_pointer);
    auto image_embed = reinterpret_cast<llava_image_embed *>(image_embed_pointer);
    auto ctx_llama = ctx_llava->ctx_llama;

//    const char * sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
//    const char * usr_prompt = "What's the main content of this photo?\nASSISTANT:";
    const char * sys_prompt = env->GetStringUTFChars(system_prompt, 0);
    const char * usr_prompt = env->GetStringUTFChars(user_prompt, 0);

    LOGi("llava_completion_init: sys_prompt: %s, usr_prompt: %s", sys_prompt, usr_prompt);
    std::vector<llama_token> sys_tokens = llama_tokenize(ctx_llama, sys_prompt, true, true);
    std::vector<llama_token> usr_tokens = llama_tokenize(ctx_llama, usr_prompt, false, true);
    LOGi("sys_tokens size: %d", sys_tokens.size());
    LOGi("usr_tokens size: %d", usr_tokens.size());

    int n_embd = llama_n_embd(llama_get_model(ctx_llama));
    LOGi("n_embd size: %d", n_embd);
    float * embds = new float [sys_tokens.size() * n_embd + usr_tokens.size() * n_embd + image_embed->n_image_pos * n_embd];
    compute_tokens_embeddings(ctx_llama, sys_tokens, embds);
    memcpy(embds + sys_tokens.size() * n_embd, image_embed->embed, image_embed->n_image_pos * n_embd * sizeof(float));
    compute_tokens_embeddings(ctx_llama, usr_tokens, embds + (sys_tokens.size() + image_embed->n_image_pos) * n_embd);
    LOGi("compute_tokens_embeddings finished");

    int32_t n_eval = sys_tokens.size() + image_embed->n_image_pos + usr_tokens.size();
    llama_batch batch = {n_eval, nullptr, embds, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, (int32_t) sys_tokens.size(), image_embed->n_image_pos, image_embed->n_image_pos/32};
    if (llama_decode(ctx_llama, batch) != 0) {
        LOGe("llava_completion_init() failed.");
    } else {
        LOGi("llava_completion_init() success!");
    }

    env->ReleaseStringUTFChars(system_prompt, sys_prompt);
    env->ReleaseStringUTFChars(user_prompt, usr_prompt);
    return reinterpret_cast<jint>(n_eval);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1completion_1loop(JNIEnv * env, jobject, jlong sampler_pointer,jlong image_embed_pointer, jlong ctx_llava_pointer, jint n_past_tokens) {
//    auto smpl = reinterpret_cast<gpt_sampler *>(sampler);
//    auto ctx_llava_ = reinterpret_cast<struct llava_context *>(ctx_llava);
//    int n_past_tokens = reinterpret_cast<int>(n_past);
//
//    const char * tmp = sample(smpl, ctx_llava_->ctx_llama, &n_past_tokens);
//
//    if (strcmp(tmp, "</s>") == 0) return nullptr;
//    jstring new_token = env->NewStringUTF(tmp);
//    return new_token;

    auto smpl = reinterpret_cast<llama_sampler*>(sampler_pointer);
    auto ctx_llava = reinterpret_cast<llava_context *>(ctx_llava_pointer);
    auto ctx_llama = ctx_llava->ctx_llama;
    auto model = ctx_llava->model;
    int n_past = reinterpret_cast<int>(n_past_tokens);

    const auto new_token_id = llama_sampler_sample(smpl, ctx_llama, -1);
    if (llama_token_is_eog(model, new_token_id)) {
        return nullptr;
    }
    auto new_tokens_chars = llama_token_to_piece(ctx_llama, new_token_id);

    std::vector<llama_token> tokens;
    tokens.push_back(new_token_id);
    llama_batch batch_one = llama_batch_get_one(&tokens[0], 1, n_past, 0);
    batch_one.img_token_step = image_embed->n_image_pos / 32;
    if (llama_decode(ctx_llama, batch_one) != 0) {
        LOGe("llava_completion_loop failed");
    }

    jstring new_token = env->NewStringUTF(new_tokens_chars.c_str());
    return new_token;

//    std::string response;
//    for (int i = 0; i < 64; i++) {
//        const auto new_token_id = llama_sampler_sample(smpl, ctx_llama, -1);
//        LOGi("new_token_id: %d", new_token_id);
//        if (llama_token_is_eog(model, new_token_id)) break;
//        auto new_tokens_chars = llama_token_to_piece(ctx_llama, new_token_id);
//        LOGi("%s", new_tokens_chars.c_str());
//        response += new_tokens_chars;
//
//        std::vector<llama_token> tokens;
//        tokens.push_back(new_token_id);
//
//        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[0], 1, n_past+i, 0)) != 0) {
//            LOGe("Fuck");
//        }
//    }
//    LOGi("response: %s", response.c_str());
//    return nullptr;
}


extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1debug(JNIEnv * env, jobject) {
    llama_model_params model_params = llama_model_default_params();
    const char * path_to_model = "/data/data/com.example.llama/files/qwen2-0_5b-instruct-q8_0.gguf";
    auto model = llama_load_model_from_file(path_to_model, model_params);
    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
    } else {
        LOGi("load_model() finished");
    }


    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 2048;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
//    ctx_params.n_ubatch        = 1024;

    llama_context * context = llama_new_context_with_model(model, ctx_params);
    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
    } else {
        LOGi("llama_new_context_with_model() finished");
    }

    int embd = 0;
    int n_tokens = 512;
    int n_seq_max = 1;
    int n_len = 2048;

    llama_batch *batch = new llama_batch {
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
            0,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    const char * text = "hello";
    const auto tokens_list = llama_tokenize(context, text, 1);

    auto n_ctx = llama_n_ctx(context);
    auto n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOGi("n_len = %d, n_ctx = %d, n_kv_req = %d", n_len, n_ctx, n_kv_req);

    for (auto id : tokens_list) {
        LOGi("%s", llama_token_to_piece(context, id).c_str());
    }

    // evaluate the initial prompt
    for (auto i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(*batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;

//    if (llama_decode(context, *batch) != 0) {
//        LOGe("llama_decode() failed");
//    } else {
//        LOGi("llama_decode() finished");
//    }

}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_llava_1debug2(JNIEnv * env, jobject) {
    llama_model_params model_params = llama_model_default_params();
//    const char * path_to_model = "/data/data/com.example.llama/files/qwen2-0_5b-instruct-q8_0.gguf";
    const char * path_to_model = "/data/data/com.example.llama/files/ggml-model-q2_k.gguf";
    auto model = llama_load_model_from_file(path_to_model, model_params);
    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
    } else {
        LOGi("load_model() finished");
    }

    const char * clip_path = "/data/data/com.example.llama/files/mmproj-model-f16.gguf";

    std::string prompt = "describe the image in detail";
    auto ctx_clip = clip_model_load(clip_path, 1);
    if (!ctx_clip) {
        LOGe("clip_model_load() failed");
    } else {
        LOGi("clip_model_load() finished");
    }


    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    LOGi("Using %d threads", n_threads);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = 2048;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_ubatch        = 1024;

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);
    if (!ctx_llama) {
        LOGe("llama_new_context_with_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_new_context_with_model() returned null)");
    } else {
        LOGi("llama_new_context_with_model() finished");
    }

    auto * ctx_llava = (struct llava_context *) malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;


    const char * path_to_image = "/data/data/com.example.llama/files/cat.jpeg";
    llava_image_embed * image_embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, n_threads, path_to_image);
    if (!image_embed) {
        LOGe("image embed failed");
    } else {
        LOGi("image embed finished");
    }

    const char * sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
    const char * usr_prompt = "What's the main content of this photo?\nASSISTANT:";
    LOGi("llava_completion_init: sys_prompt: %s, usr_prompt: %s", sys_prompt, usr_prompt);
    std::vector<llama_token> sys_tokens = llama_tokenize(ctx_llama, sys_prompt, true, true);
    std::vector<llama_token> usr_tokens = llama_tokenize(ctx_llama, usr_prompt, false, true);
    LOGi("sys_tokens size: %d", sys_tokens.size());
    LOGi("usr_tokens size: %d", usr_tokens.size());

//    int n_embd = llama_n_embd(llama_get_model(ctx_llama));
//    LOGi("n_embd size: %d", n_embd);
//    float * embds = new float [sys_tokens.size() * n_embd + usr_tokens.size() * n_embd];
//    compute_tokens_embeddings(ctx_llama, sys_tokens, embds);
//    compute_tokens_embeddings(ctx_llama, usr_tokens, embds + sys_tokens.size() * n_embd);

    int n_embd = llama_n_embd(llama_get_model(ctx_llama));
    LOGi("n_embd size: %d", n_embd);
    float * embds = new float [sys_tokens.size() * n_embd + usr_tokens.size() * n_embd + image_embed->n_image_pos * n_embd];
    compute_tokens_embeddings(ctx_llama, sys_tokens, embds);
    memcpy(embds + sys_tokens.size() * n_embd, image_embed->embed, image_embed->n_image_pos * n_embd * sizeof(float));
    compute_tokens_embeddings(ctx_llama, usr_tokens, embds + (sys_tokens.size() + image_embed->n_image_pos) * n_embd);

    int32_t n_eval = sys_tokens.size() + image_embed->n_image_pos + usr_tokens.size();
    llama_batch batch = {n_eval, nullptr, embds, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, (int32_t) sys_tokens.size(), image_embed->n_image_pos, image_embed->n_image_pos/32};

    LOGi("compute_tokens_embeddings finished");

//    int32_t n_eval = sys_tokens.size() + usr_tokens.size();
//    llama_batch batch = {n_eval, nullptr, embds, nullptr, nullptr, nullptr, nullptr, 0, 1, 0};
    if (llama_decode(ctx_llama, batch) != 0) {
        LOGe("Fuck");
    } else {
        LOGi("llama_decode success!");
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

//    const auto new_token_id = llama_sampler_sample(smpl, ctx_llama, -1);
//    auto new_tokens_chars = llama_token_to_piece(ctx_llama, new_token_id);
//    LOGi("%s", new_tokens_chars.c_str());

    std::string response;

    int n_past = n_eval;
    for (int i = 0; i < 64; i++) {
//        const char * tmp = sample(smpl, ctx_llama, &n_past);
//        if (strcmp(tmp, "</s>") == 0) break;
//        LOGi("%s", tmp);
        const auto new_token_id = llama_sampler_sample(smpl, ctx_llama, -1);
        LOGi("new_token_id: %d", new_token_id);
        if (llama_token_is_eog(model, new_token_id)) break;
        auto new_tokens_chars = llama_token_to_piece(ctx_llama, new_token_id);
        LOGi("%s", new_tokens_chars.c_str());
        response += new_tokens_chars;

        std::vector<llama_token> tokens;
        tokens.push_back(new_token_id);

        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[0], 1, n_past+i, 0)) != 0) {
            LOGe("Fuck");
        }
    }

    LOGi("response: %s", response.c_str());
    delete [] embds;

}
