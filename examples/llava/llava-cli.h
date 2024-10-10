#ifndef LLAVA_CLI_H
#define LLAVA_CLI_H

#include "clip.h"
#include "common.h"
#include "llama.h"
#include "llava.h"

#include <string>



#ifdef __cplusplus
extern "C" {
#endif

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

void compute_tokens_embeddings(struct llama_context * ctx_llama, std::vector<llama_token> & tokens, float * embds);

const char * sample(struct gpt_sampler * smpl, struct llama_context * ctx_llama, int * n_past);

bool eval_text_img_prompt(struct llama_context * ctx_llama, struct llava_image_embed * image_embed, const char * str1, const char * str2, int * n_past);

struct llava_image_embed * load_image(llava_context * ctx_llava, gpt_params * params, const std::string & fname);

struct llama_model * llava_init(gpt_params * params);

struct llava_context * llava_init_context(gpt_params * params, struct llama_model * model);

void llava_free(struct llava_context * ctx_llava);

#ifdef __cplusplus
}
#endif

#endif

