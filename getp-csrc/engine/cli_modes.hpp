// Entry points that main() in run.cpp reaches through the macros at the end of
// getp_run.cpp. The checkpoint argument may be an exported .bin file (handled
// by run.cpp, unchanged) or a Hugging Face model folder (handled here).
#pragma once

#include "cpu_forward.hpp"

// Forward pass for one token of one sequence, for whichever weights are loaded.
static float *model_forward(Transformer *transformer, int token, int pos) {
  if (g_hf.loaded)
    return hf_cpu_forward(transformer, token, pos);
  return forward(transformer, token, pos);
}

static void gptoss_build_transformer(Transformer *t, char *checkpoint_path) {
  if (hf_is_dir(checkpoint_path))
    hf_build_transformer(t, checkpoint_path);
  else
    build_transformer(t, checkpoint_path);
}

static void gptoss_free_transformer(Transformer *t) {
  free_transformer(t);
  if (g_hf.loaded)
    hf_free();
}

// generate() from run.cpp, calling model_forward() instead of forward().
static void gptoss_generate(Transformer *transformer, Tokenizer *tokenizer,
                            Sampler *sampler, const char *prompt, int steps) {
  const char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
  encode(tokenizer, prompt, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  long start = 0;
  int next;
  int token = prompt_tokens[0];
  int pos = 0;

  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {
    float *logits = model_forward(transformer, token, pos);

    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      next = sample(sampler, logits);
    }
    pos++;

    // EOS (=199999 or =200002) ends the sequence
    if (next == 199999 || next == 200002) {
      break;
    }

    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);
    token = next;

    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n",
            (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

// chat() from run.cpp, calling model_forward() instead of forward().
static void gptoss_chat(Transformer *transformer, Tokenizer *tokenizer,
                        Sampler *sampler, const char *cli_user_prompt,
                        const char *cli_system_prompt, int steps) {
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
  int user_idx;

  int8_t user_turn = 1;
  int next;
  int token;
  int pos = 0;
  while (pos < steps) {

    if (user_turn) {
      if (pos == 0) {
        if (cli_system_prompt == NULL) {
          read_stdin("Enter system prompt (optional): ", system_prompt,
                     sizeof(system_prompt));
        } else {
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      if (pos == 0 && cli_user_prompt != NULL) {
        strcpy(user_prompt, cli_user_prompt);
      } else {
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      encode(tokenizer, rendered_prompt, -1, -1, prompt_tokens,
             &num_prompt_tokens, transformer->config.initial_context_length);
      user_idx = 0;
      user_turn = 0;
      printf("Assistant: ");
    }

    if (user_idx < num_prompt_tokens) {
      token = prompt_tokens[user_idx++];
    } else {
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) {
      user_turn = 1;
    }

    float *logits = model_forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      const char *piece = decode_piece(tokenizer, token, next);
      safe_printf(piece);
      fflush(stdout);
    }
    if (next == 2) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
}
