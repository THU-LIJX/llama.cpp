#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "sampling.h"
#include "clip.h"
#include "llava-cli.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch_one = llama_batch_get_one(&tokens[i], n_eval, *n_past, 0);
        batch_one.img_token_step = 4;// 需要修改一下实际值
        if (llama_decode(ctx_llama, batch_one)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

const char * sample(struct gpt_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = gpt_sampler_sample(smpl, ctx_llama, -1);
    gpt_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

struct llava_image_embed * load_image(llava_context * ctx_llava, gpt_params * params, const std::string & fname) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            LOG_INF("using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->cpuparams.n_threads, prompt);
        if (!embed) {
            LOG_ERR("%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->cpuparams.n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    return embed;
}

void compute_tokens_embeddings(struct llama_context * ctx_llama, std::vector<llama_token> & tokens, float * embds) {
    llama_batch batch = llama_batch_get_one(&tokens[0], tokens.size(), 0, 0);

    int n_embd = llama_n_embd(llama_get_model(ctx_llama));
    // n_tokens * n_embd * sizeof(float)

    struct ggml_init_params params {
        /* .mem_size = */ tokens.size() * n_embd * sizeof(float) * 10,
        /* .mem_buffer = */ NULL,
        /* .no_alloc = */ false,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens);
    memcpy(inp_tokens->data, &tokens[0], batch.n_tokens * sizeof(int32_t));

    // TODO: delete debug code
    // struct ggml_tensor * inp_arange_drop = ggml_arange_drop(ctx0, inp_tokens, 0, 1000, 2, 0);

    // ggml_build_forward_expand(gf, inp_arange_drop);
    // ggml_graph_compute_with_ctx(ctx0, gf, 1);
    // end


    struct ggml_tensor * tok_embd = llama_get_model_tok_embd(llama_get_model(ctx_llama));
    struct ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    struct ggml_tensor * inp_embd_flat = ggml_view_1d(ctx0, inp_embd, n_embd * batch.n_tokens, 0);


    // struct ggml_tensor * inp_embd_flat_5 = ggml_view_1d(ctx0, inp_embd_flat, 5, 0);
    // struct ggml_tensor * inp_embd_flat_order = ggml_argsort(ctx0, inp_embd_flat_5, GGML_SORT_ORDER_ASC);
    // ggml_build_forward_expand(gf, inp_embd_flat_order);
    // struct ggml_tensor * inp_embd_flat_topk = ggml_top_k(ctx0, inp_embd_flat_5, 3);
    // ggml_build_forward_expand(gf, inp_embd_flat_topk);

    ggml_build_forward_expand(gf, inp_embd_flat);

    ggml_graph_compute_with_ctx(ctx0, gf, 1);
    struct ggml_tensor * result = ggml_graph_node(gf, -1);

    memcpy(embds, result->data, n_embd * batch.n_tokens * sizeof(float));
    ggml_free(ctx0);
}


bool eval_text_img_prompt(struct llama_context * ctx_llama, struct llava_image_embed * image_embed, const char * str1, const char * str2, int * n_past) {
    std::string system_prompt = str1;
    std::string user_prompt = str2;
    std::vector<llama_token> sys_tokens = ::llama_tokenize(ctx_llama, system_prompt, true, true);
    std::vector<llama_token> user_tokens = ::llama_tokenize(ctx_llama, user_prompt, false, true);

    int n_embd = llama_n_embd(llama_get_model(ctx_llama));
    float * embds = new float[sys_tokens.size() * n_embd + user_tokens.size() * n_embd + image_embed->n_image_pos * n_embd];
    compute_tokens_embeddings(ctx_llama, sys_tokens, embds);
    memcpy(embds + sys_tokens.size() * n_embd, image_embed->embed, image_embed->n_image_pos * n_embd * sizeof(float));
    compute_tokens_embeddings(ctx_llama, user_tokens, embds + (sys_tokens.size() + image_embed->n_image_pos) * n_embd);

    int32_t n_eval = sys_tokens.size() + image_embed->n_image_pos + user_tokens.size();
    // TODO: 这里假设了模型的层数为32
    llama_batch batch = {n_eval, nullptr, embds, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, (int32_t) sys_tokens.size(), image_embed->n_image_pos, image_embed->n_image_pos/32};
    // llama_batch batch = {sys_tokens.size() + image_embed->n_image_pos + user_tokens.size(), nullptr, embds, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, sys_tokens.size(), image_embed->n_image_pos, 1};
    if (llama_decode(ctx_llama, batch)) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }
    *n_past += sys_tokens.size() + image_embed->n_image_pos + user_tokens.size();
    return true;
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, gpt_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    // eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    // llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    // eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
    eval_text_img_prompt(ctx_llava->ctx_llama, image_embed, system_prompt.c_str(), user_prompt.c_str(), &n_past);

    // generate the response

    LOG("\n");

    struct gpt_sampler * smpl = gpt_sampler_init(ctx_llava->model, params->sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    gpt_sampler_free(smpl);
    LOG("\n");
}

struct llama_model * llava_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

struct llava_context * llava_init_context(gpt_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);


    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

// llava_init()
// llava_init_context()
// process_prompt()
// llava_free()

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    gpt_init();

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }

    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt)) {
        auto * ctx_llava = llava_init_context(&params, model);

        auto * image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    } else {
        for (auto & image : params.image) {
            auto * ctx_llava = llava_init_context(&params, model);

            auto * image_embed = load_image(ctx_llava, &params, image);
            if (!image_embed) {
                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);

            llama_perf_context_print(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_free_model(model);

    return 0;
}
